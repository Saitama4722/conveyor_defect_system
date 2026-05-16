"""Конвертация MVTec-масок в YOLO OBB-разметку (мультиклассовая версия)."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np


DATASET_ROOT: Path = Path("datasets/dataset/dataset")
OUTPUT_ROOT: Path = Path("datasets/yolo_dataset_multiclass")
TRAIN_RATIO: float = 0.8
RANDOM_SEED: int = 42
MIN_MASK_AREA: int = 100   # минимальная площадь дефекта в пикселях

# Имя папки дефекта → индекс класса
CLASS_MAP: dict[str, int] = {
    "bent": 0,
    "scratch": 1,
    "scratch_head": 1,
    "scratch_neck": 1,
    "color": 2,
    "broken_large": 3,
    "broken_small": 4,
    "contamination": 5,
    "thread_side": 6,
    "thread_top": 7,
}

# Имена классов в порядке индексов 0..7
CLASS_NAMES: list[str] = [
    "bent", "scratch", "color",
    "broken_large", "broken_small", "contamination",
    "thread_side", "thread_top",
]

# Категория → список папок дефектов
CATEGORIES: dict[str, list[str]] = {
    "metal_nut": ["bent", "scratch", "color"],
    "screw": ["scratch_head", "scratch_neck", "thread_side", "thread_top"],
    "bottle": ["broken_large", "broken_small", "contamination"],
}


def mask_to_obb(mask_path: Path) -> list[list[float]] | None:
    """Из бинарной маски получить список OBB в нормированных координатах."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    h, w = mask.shape[:2]
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obbs: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_MASK_AREA:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)  # (4, 2)
        flat: list[float] = []
        for x, y in box:
            flat.append(float(x) / w)
            flat.append(float(y) / h)
        obbs.append(flat)

    return obbs if obbs else None


def convert_category(
    category: str,
    defect_type: str,
    class_idx: int,
    mask_paths: list[Path],
    output_images: Path,
    output_labels: Path,
    split: str,
) -> int:
    """Конвертировать переданные маски одной пары (категория, дефект) в split."""
    images_dir = DATASET_ROOT / category / "test" / defect_type

    if not images_dir.exists():
        print(f"[пропуск] нет каталога изображений: {images_dir}")
        return 0

    count = 0
    for mask_path in mask_paths:
        stem = mask_path.stem.replace("_mask", "")
        image_path = images_dir / f"{stem}.png"
        if not image_path.exists():
            print(f"[пропуск] нет исходного изображения: {image_path}")
            continue

        obbs = mask_to_obb(mask_path)
        if not obbs:
            continue

        # Префикс предотвращает коллизии имён между категориями.
        out_stem = f"{category}_{defect_type}_{stem}"
        out_image = output_images / f"{out_stem}.png"
        out_label = output_labels / f"{out_stem}.txt"

        shutil.copy2(image_path, out_image)
        with out_label.open("w", encoding="utf-8") as f:
            for obb in obbs:
                coords = " ".join(f"{v:.6f}" for v in obb)
                f.write(f"{class_idx} {coords}\n")

        count += 1

    if count:
        print(f"[{split}] {category}/{defect_type} (cls={class_idx}): {count} изображений")
    return count


def _write_data_yaml(path: Path) -> None:
    """Записать data.yaml для ultralytics."""
    names_block = ", ".join(CLASS_NAMES)
    content = (
        f"path: {OUTPUT_ROOT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names_block}]\n"
    )
    path.write_text(content, encoding="utf-8")


def convert_all() -> None:
    """Главный конвейер: создать структуру и обработать все триплеты."""
    rng = random.Random(RANDOM_SEED)

    images_train = OUTPUT_ROOT / "images" / "train"
    images_val = OUTPUT_ROOT / "images" / "val"
    labels_train = OUTPUT_ROOT / "labels" / "train"
    labels_val = OUTPUT_ROOT / "labels" / "val"
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0
    # Разбиение 80/20 выполняется внутри каждой подпапки дефекта,
    # чтобы каждый класс присутствовал и в train, и в val.
    for category, defects in CATEGORIES.items():
        for defect_type in defects:
            class_idx = CLASS_MAP[defect_type]
            masks_dir = DATASET_ROOT / category / "ground_truth" / defect_type
            if not masks_dir.exists():
                print(f"[пропуск] нет каталога масок: {masks_dir}")
                continue

            mask_paths = sorted(masks_dir.glob("*_mask.png"))
            rng.shuffle(mask_paths)
            split_at = int(len(mask_paths) * TRAIN_RATIO)
            train_masks = mask_paths[:split_at]
            val_masks = mask_paths[split_at:]

            train_count += convert_category(
                category, defect_type, class_idx, train_masks,
                images_train, labels_train, "train",
            )
            val_count += convert_category(
                category, defect_type, class_idx, val_masks,
                images_val, labels_val, "val",
            )

    _write_data_yaml(OUTPUT_ROOT / "data.yaml")

    total = train_count + val_count
    print("---")
    print(f"Всего изображений: {total}")
    print(f"  train: {train_count}")
    print(f"  val:   {val_count}")


if __name__ == "__main__":
    convert_all()
