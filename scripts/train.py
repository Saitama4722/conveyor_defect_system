"""Обучение YOLOv8n-OBB на подготовленном датасете дефектов."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


DATASET_YAML: str = r"F:\YandexDisk\Работа\studlance\925750\f\conveyor_defect_system\datasets\yolo_dataset_multiclass\data.yaml"
MODEL_BASE: str = "yolov8n-obb.pt"
OUTPUT_DIR: str = "runs/train"
EXPERIMENT_NAME: str = "defect_detector_obb"
EPOCHS: int = 200
IMG_SIZE: int = 640
BATCH_SIZE: int = 8
PATIENCE: int = 50
LR0: float = 0.01
LRF: float = 0.01
MOMENTUM: float = 0.937
WEIGHT_DECAY: float = 0.0005
CONF_THRESHOLD: float = 0.5
IOU_THRESHOLD: float = 0.45
DEVICE: str = "0" if torch.cuda.is_available() else "cpu"

FINAL_MODEL_PATH: Path = Path("detection/model/best.pt")


def train() -> Path:
    """Запустить обучение и скопировать лучший чекпойнт в detection/model/."""
    model = YOLO(MODEL_BASE)
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        project=OUTPUT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    best_path = Path(OUTPUT_DIR) / EXPERIMENT_NAME / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Лучший чекпойнт не найден: {best_path}")

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, FINAL_MODEL_PATH)
    print(f"Сохранена модель: {FINAL_MODEL_PATH}")
    return best_path


def validate(model_path: str) -> dict:
    """Прогнать валидацию обученной модели и вернуть метрики."""
    model = YOLO(model_path)
    results = model.val(
        data=DATASET_YAML,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
    )

    metrics = getattr(results, "results_dict", None)
    if isinstance(metrics, dict):
        return dict(metrics)
    return {}


if __name__ == "__main__":
    best_path = train()
    metrics = validate(str(best_path))
    print(f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
