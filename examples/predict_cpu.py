"""Run a minimal YOLO26 detection and print the public result contract."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".local" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".local" / "cache"))

from ultralytics import YOLO  # noqa: E402


MODEL_PATH = ROOT / "yolo26n.pt"
IMAGE_PATH = ROOT / "data" / "bus.jpg"


def main() -> None:
    model = YOLO(str(MODEL_PATH))
    result = model.predict(
        source=str(IMAGE_PATH),
        device="cpu",
        imgsz=640,
        conf=0.25,
        save=False,
        verbose=False,
    )[0]

    boxes = result.boxes
    print(f"predictor: {type(model.predictor).__module__}.{type(model.predictor).__name__}")
    print(f"result: {type(result).__module__}.{type(result).__name__}")
    print(f"original image: shape={result.orig_img.shape}, dtype={result.orig_img.dtype}")
    print(f"boxes.data: shape={tuple(boxes.data.shape)}, dtype={boxes.data.dtype}, device={boxes.data.device}")
    print(f"speed (ms): {result.speed}")

    for index, (xyxy, confidence, class_id) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls), start=1):
        class_index = int(class_id)
        print(
            f"detection {index}: class={result.names[class_index]!r}, "
            f"confidence={float(confidence):.3f}, xyxy={[round(float(value), 1) for value in xyxy]}"
        )


if __name__ == "__main__":
    main()
