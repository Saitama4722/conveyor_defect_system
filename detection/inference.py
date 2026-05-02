"""YOLOv8n-OBB инференс для детекции дефектов на конвейере."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from detection.ghost_conv import replace_conv_with_ghost


MODEL_PATH_DEFAULT: str = "detection/model/best.pt"
CONF_THRESHOLD_DEFAULT: float = 0.3
IOU_THRESHOLD: float = 0.45
IMG_SIZE: int = 640
DEVICE_AUTO: str = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "bent":          (0,   0,   255),
    "scratch":       (0,   165, 255),
    "color":         (0,   255, 255),
    "broken_large":  (0,   255, 0  ),
    "broken_small":  (255, 165, 0  ),
    "contamination": (255, 0,   0  ),
    "thread_side":   (255, 0,   255),
    "thread_top":    (128, 0,   255),
    "good":          (200, 200, 200),
}

DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)
TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.7
FONT_THICKNESS: int = 2
POLY_THICKNESS: int = 3


class DefectDetector:
    """Обёртка над YOLOv8n-OBB для детекции дефектов."""

    def __init__(
        self,
        model_path: str = MODEL_PATH_DEFAULT,
        conf: float = CONF_THRESHOLD_DEFAULT,
        apply_ghost: bool = False,
    ) -> None:
        self.model_path: str = model_path
        self.conf: float = conf
        self.apply_ghost: bool = apply_ghost
        self.device: str = DEVICE_AUTO
        self.model: YOLO | None = None
        self.load_model()

    def load_model(self) -> None:
        """Загрузить веса YOLO и (опционально) подменить свёртки на GhostConv."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

        self.model = YOLO(self.model_path)
        if self.apply_ghost:
            replace_conv_with_ghost(self.model.model)
        self.model.to(self.device)

    def predict(self, frame: np.ndarray) -> list[dict]:
        """Запустить инференс на одном BGR-кадре."""
        if self.model is None:
            return []

        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        obb = getattr(result, "obb", None)
        if obb is None or obb.xyxyxyxy is None:
            return []

        polys = obb.xyxyxyxy.cpu().numpy()    # (N, 4, 2)
        clses = obb.cls.cpu().numpy().astype(int)
        confs = obb.conf.cpu().numpy()

        detections: list[dict] = []
        for poly, cls_idx, conf in zip(polys, clses, confs):
            points = poly.astype(np.float32).reshape(4, 2)
            xs = points[:, 0]
            ys = points[:, 1]
            xyxy = (
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()),
            )
            detections.append(
                {
                    "class_name": names[int(cls_idx)],
                    "confidence": float(conf),
                    "points": points,
                    "xyxy": xyxy,
                }
            )
        return detections

    def draw_results(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Нарисовать OBB-контуры и подписи поверх кадра."""
        # одноразовая диагностика
        if not hasattr(self, '_draw_debug_done'):
            self._draw_debug_done = True
            print(f"[DRAW] detections count: {len(detections)}")
            if detections:
                print(f"[DRAW] first detection: {detections[0]}")

        vis = frame.copy()

        # зелёный для всех боксов; чёрный текст на зелёном фоне
        color = (0, 255, 0)
        label_color = (0, 255, 0)
        text_color = (0, 0, 0)

        for det in detections:
            class_name: str = det["class_name"]
            confidence: float = det["confidence"]

            # координаты YOLO уже в пиксельном пространстве кадра
            pts = det["points"].astype(int)

            if pts.ndim == 2 and pts.shape == (4, 2):
                cv2.polylines(vis, [pts.reshape((-1, 1, 2))],
                              True, color, thickness=3)

            label = f"{class_name} {confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

            anchor_x = int(pts[:, 0].min())
            anchor_y = int(pts[:, 1].min())
            top_left = (anchor_x, max(0, anchor_y - th - baseline - 2))
            bottom_right = (anchor_x + tw + 2, anchor_y)
            cv2.rectangle(vis, top_left, bottom_right, label_color, thickness=-1)
            cv2.putText(
                vis,
                label,
                (anchor_x + 1, anchor_y - baseline - 1),
                FONT,
                FONT_SCALE,
                text_color,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return vis

    def get_stats(self, detections: list[dict]) -> dict:
        """Сводка по детекциям: всего и по классам."""
        by_class = Counter(det["class_name"] for det in detections)
        return {
            "total": len(detections),
            "by_class": dict(by_class),
        }
