# YOLO26 inference research sandbox

Минимальное окружение для изучения preprocessing, inference, postprocessing и tracking в фактически зафиксированной версии Ultralytics. Это исследовательский эталон, а не продуктовая реализация.

## Окружение

- Python 3.10.12
- Ultralytics 8.4.115
- PyTorch 2.13.0
- torchvision 0.28.0
- OpenCV 5.0.0
- `yolo26n.pt`, detect, CPU

Установка без зависимости от активации shell:

```bash
.venv/bin/python -m pip install -r requirements-research.txt
```

## Подготовка локальных данных

Веса и тестовые данные намеренно не хранятся в Git. Для примеров нужны:

- `yolo26n.pt` в корне проекта;
- `data/bus.jpg`;
- `data/people-counting.mp4`.

Используются публичные assets Ultralytics. Не добавляйте клиентские изображения, видео или логи.

## Запуск

Один image через detection pipeline:

```bash
.venv/bin/python examples/predict_cpu.py
```

Первые 20 кадров через tracking pipeline:

```bash
.venv/bin/python examples/track_cpu.py --frames 20
```

Скрипты ничего не визуализируют и не сохраняют в `runs/`: они печатают классы runtime-объектов, shapes, dtype, device, timings и tracking IDs.

## Наблюдённый runtime-маршрут

```text
YOLO
  -> DetectionPredictor
  -> preprocessing
  -> PyTorch model on CPU
  -> detection postprocessing
  -> Results
  -> Boxes

track():
  -> тот же DetectionPredictor
  -> tracker callbacks/state
  -> TRACKTRACK (tracktrack.yaml по умолчанию)
  -> Boxes с track IDs
```

Следующий этап исследования — поставить небольшие probes на границы preprocessing/inference/postprocessing и зафиксировать тип, shape, dtype, диапазон значений и device для одного кадра.

## Лицензионная граница

Ultralytics и официальные веса используются только как исследовательский эталон. Не копируйте и не переносите исходный код библиотеки в независимую реализацию. Применимость AGPL-3.0, Enterprise license и лицензии весов к продукту должна быть отдельно подтверждена ответственными за legal/compliance.
