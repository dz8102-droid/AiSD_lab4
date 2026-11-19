#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа 4. Дообучение YOLOv8s для детекции квадрокоптеров (дронов).
Скрипт ОБУЧЕНИЯ (fine-tuning).

Запускается ПОСЛЕ подготовки датасета: структура папок train/val/test,
файл data.yaml и разметка в YOLO-формате (txt рядом с фото).

Установка зависимостей:
    pip install ultralytics

Пример запуска:
    python train.py --data data.yaml --epochs 50 --batch 16

На компьютере без GPU укажите: --device cpu
Результаты (веса best.pt + графики кривых) сохраняются в runs/detect/<name>/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Дообучение YOLOv8s на детекцию квадрокоптеров (дронов)")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="предобученные веса (по умолчанию yolov8s.pt, COCO)")
    parser.add_argument("--data", default="data.yaml",
                        help="путь к конфигу датасета data.yaml")
    parser.add_argument("--epochs", type=int, default=50,
                        help="число эпох обучения (по умолчанию 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="размер входного изображения (по умолчанию 640)")
    parser.add_argument("--batch", type=int, default=8,        # ← было 16
                        help="размер батча; при ошибке OOM уменьшите до 4")
    parser.add_argument("--device", default="cpu",             # ← было "0"
                        help="устройство: 0 — первый GPU, cpu — процессор")
    parser.add_argument("--name", default="yolov8s_drone_v1",
                        help="имя эксперимента (папка runs/detect/<name>)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Загружаем предобученную модель (веса COCO скачаются автоматически).
    #    Модель уже умеет находить 80 общих классов, мы только «объясняем»
    #    ей новый класс — поэтому 100-150 фото и 50 эпох обычно достаточно.
    model = YOLO(args.model)

    # 2. Запускаем дообучение.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=15,      # ранняя остановка: 15 эпох без улучшения метрики
        # --- Аугментации (встроены в YOLOv8; значения подобраны под задачу) ---
        hsv_h=0.015,      # сдвиг оттенка — имитация изменения освещения
        hsv_s=0.7,        # сдвиг насыщенности (контраст дрона на фоне неба)
        hsv_v=0.4,        # сдвиг яркости (тень / сумерки)
        degrees=5.0,      # небольшие повороты изображения
        translate=0.1,    # случайные сдвиги по осям
        scale=0.5,        # масштабирование (ближние/дальние дроны)
        fliplr=0.5,       # горизонтальное отражение с вероятностью 0.5
        mosaic=1.0,       # мозаика из 4 изображений — учитывает мелкие объекты (дрон в небе!)
    )

    # 3. Валидация ЛУЧШЕЙ модели на val-выборке: печатаем метрики для отчёта.
    # Автоматический поиск последней папки с весами
    runs_dir = Path("runs/detect")
    if runs_dir.exists():
        # Берем самую свежую папку
        last_run = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        best_weights = last_run / "weights" / "best.pt"
    else:
        best_weights = Path("runs/detect") / args.name / "weights" / "best.pt"
    model = YOLO(str(best_weights))
    metrics = model.val(data=args.data)

    print("\n" + "=" * 52)
    print("МЕТРИКИ ДЛЯ ОТЧЁТА (валидация на val):")
    print(f"  mAP@50    : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95 : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print("=" * 52)
    print(f"Лучшие веса сохранены в: {best_weights}")
    print(f"Кривые обучения: runs/detect/{args.name}/results.png")
    print(f"Матрица ошибок: runs/detect/{args.name}/confusion_matrix.png")


if __name__ == "__main__":
    main()
