#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа 4. Скрипт ИНФЕРЕНСА: детекция квадрокоптеров (дронов).

Прогоняет тестовые изображения (или видео) через дообученную модель,
сохраняет картинки с отрисованными рамками и печатает текстовую сводку
с уровнем уверенности (confidence) каждого обнаружения.

Установка зависимостей:
    pip install ultralytics

Пример запуска:
    python inference.py --weights runs/detect/yolov8s_drone_v1/weights/best.pt \
                        --source test_images --conf 0.4

После запуска размеченные картинки появятся в runs/predict/test_results/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Инференс YOLOv8s: детекция квадрокоптеров (дронов)")
    parser.add_argument("--weights", required=True,
                        help="путь к дообученным весам best.pt")
    parser.add_argument("--source", required=True,
                        help="тестовые данные: папка, отдельное фото или видео")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="порог уверенности (по умолчанию 0.4)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="размер входного изображения (по умолчанию 640)")
    parser.add_argument("--device", default="cpu",             # ← было "0"
                        help="устройство: 0 — первый GPU, cpu — процессор")
    parser.add_argument("--name", default="test_results",
                        help="имя папки с результатами (runs/predict/<name>)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Загружаем ДООБУЧЕННУЮ модель (best.pt, а не yolov8s.pt!)
    model = YOLO(args.weights)

    # 2. Прогоняем инференс. save=True сохраняет изображения
    #    с отрисованными рамками, подписями класса и confidence.
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project="runs/predict",
        name=args.name,
        line_width=2,
    )

    # 3. Текстовая сводка: что и с какой уверенностью нашлось на каждом файле.
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ДЕТЕКЦИИ:")
    total = 0
    for r in results:
        image_name = Path(r.path).name
        n = len(r.boxes)
        total += n
        print(f"  {image_name}: найдено объектов — {n}")
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"      {cls_name}: confidence {conf:.2f}")
    print("=" * 60)
    print(f"Всего обнаружено: {total}")
    print(f"Картинки с рамками сохранены в: runs/predict/{args.name}/")


if __name__ == "__main__":
    main()
