"""Конвертация MVTec-масок в YOLO OBB-разметку."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np


DATASET_ROOT: Path = Path("datasets/dataset")
OUTPUT_ROOT: Path = Path("datasets/yolo_dataset")
CATEGORIES: list[str] = ["metal_nut", "screw", "bottle"]
TRAIN_RATIO: float = 0.8
RANDOM_SEED: int = 42
MIN_MASK_AREA: int = 100   # минимальная площадь дефекта в пикселях

# Глобальные имена классов в порядке индексов 0..7
CLASS_NAMES: list[str] = [
    "bent",            # 0
    "scratch",         # 1
    "color",           # 2
    "broken_large",    # 3
    "broken_small",    # 4
    "contamination",   # 5
    "thread_side",     # 6
    "thread_top",      # 7
]

# Соответствие (категория → дефект → индекс класса)
CLASS_MAP: dict[str, dict[str, int]] = {
    "metal_nut": {
        "bent": 0,
        "scratch": 1,
        "color": 2,
    },
    "screw": {
        "scratch": 1,
        "thread_side": 6,
        "thread_top": 7,
    },
    "bottle": {
        "broken_large": 3,
        "broken_small": 4,
        "contamination": 5,
    },
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
    output_images: Path,
    output_labels: Path,
    split: str,
) -> int:
    """Конвертировать одну пару (категория, дефект) в указанный split."""
    masks_dir = DATASET_ROOT / category / "ground_truth" / defect_type
    images_dir = DATASET_ROOT / category / "test" / defect_type

    if not masks_dir.exists():
        print(f"[пропуск] нет каталога масок: {masks_dir}")
        return 0
    if not images_dir.exists():
        print(f"[пропуск] нет каталога изображений: {images_dir}")
        return 0

    count = 0
    for mask_path in sorted(masks_dir.glob("*_mask.png")):
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
        print(f"[{split}] {category}/{defect_type}: {count} изображений")
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

    # Делим на train/val на уровне триплетов (категория, дефект)
    triplets: list[tuple[str, str, int]] = []
    for category in CATEGORIES:
        for defect_type, class_idx in CLASS_MAP.get(category, {}).items():
            triplets.append((category, defect_type, class_idx))

    rng.shuffle(triplets)
    split_at = int(len(triplets) * TRAIN_RATIO)
    train_triplets = triplets[:split_at]
    val_triplets = triplets[split_at:]

    train_count = 0
    for category, defect_type, class_idx in train_triplets:
        train_count += convert_category(
            category, defect_type, class_idx, images_train, labels_train, "train"
        )

    val_count = 0
    for category, defect_type, class_idx in val_triplets:
        val_count += convert_category(
            category, defect_type, class_idx, images_val, labels_val, "val"
        )

    _write_data_yaml(OUTPUT_ROOT / "data.yaml")

    total = train_count + val_count
    print("---")
    print(f"Всего изображений: {total}")
    print(f"  train: {train_count}")
    print(f"  val:   {val_count}")


if __name__ == "__main__":
    convert_all()
