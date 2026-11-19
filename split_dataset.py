#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа 4. Разбиение датасета на train / val / test (80/10/10).

На входе: папка с фото + папка с txt-разметкой (YOLO-формат, экспорт из CVAT).
На выходе: структура, которую ожидает YOLOv8:

    datasets/drones/
        images/train  images/val  images/test
        labels/train  labels/val  labels/test

Установка зависимостей: не требуются (только стандартная библиотека).

Пример запуска:
    python split_dataset.py --src-images raw/images --src-labels raw/labels \
                            --dst datasets/drones
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}  # доли выборок
SEED = 42  # фиксированный seed для воспроизводимости разбиения


def parse_args():
    parser = argparse.ArgumentParser(
        description="Разбиение датасета YOLO на train/val/test (80/10/10)")
    parser.add_argument("--src-images", required=True,
                        help="папка с исходными фото")
    parser.add_argument("--src-labels", required=True,
                        help="папка с txt-файлами разметки YOLO")
    parser.add_argument("--dst", default="datasets/drones",
                        help="корень итоговой структуры (по умолчанию datasets/drones)")
    return parser.parse_args()


def main():
    args = parse_args()
    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    dst = Path(args.dst)

    # 1. Собираем список фото и проверяем, что для каждого есть разметка
    images = sorted(p for p in src_images.iterdir()
                    if p.suffix.lower() in IMG_EXT)
    missing = [p.name for p in images if not (src_labels / f"{p.stem}.txt").exists()]
    if missing:
        print(f"[!] ВНИМАНИЕ: для {len(missing)} фото нет txt-разметки, "
              f"они будут пропущены. Примеры: {missing[:5]}")
        images = [p for p in images if p.name not in set(missing)]

    if len(images) < 20:
        print(f"[!] Мало фото ({len(images)}). Для лабораторной нужно 100-150.")
    print(f"Всего фото с разметкой: {len(images)}")

    # 2. Перемешиваем (фиксированный seed = одинаковое разбиение при повторе)
    random.seed(SEED)
    random.shuffle(images)

    # 3. Делим: 80% train, 10% val, 10% test
    n = len(images)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    # 4. Копируем фото и соответствующие txt-файлы в новую структуру
    for split, files in splits.items():
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, img_dir / img.name)
            label = src_labels / f"{img.stem}.txt"
            if label.exists():
                shutil.copy(label, lbl_dir / label.name)

    print(f"train: {len(splits['train'])}  |  val: {len(splits['val'])}  |  "
          f"test: {len(splits['test'])}")
    print(f"Готово. Структура создана в: {dst.resolve()}")
    print("Теперь пропишите путь к этой папке в data.yaml (поле path).")


if __name__ == "__main__":
    main()
