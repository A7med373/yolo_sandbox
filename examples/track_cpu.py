"""Run a short YOLO26 tracking stream and print the public tracker contract."""

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".local" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".local" / "cache"))

from ultralytics import YOLO  # noqa: E402


MODEL_PATH = ROOT / "yolo26n.pt"
VIDEO_PATH = ROOT / "data" / "people-counting.mp4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()

    model = YOLO(str(MODEL_PATH))
    results = model.track(
        source=str(VIDEO_PATH),
        device="cpu",
        imgsz=640,
        conf=0.25,
        save=False,
        stream=True,
        verbose=False,
    )

    frames_seen = 0
    frames_with_ids = 0
    track_ids: set[int] = set()
    sample_track: dict[str, object] | None = None

    for frames_seen, result in enumerate(results, start=1):
        boxes = result.boxes
        if boxes.id is not None:
            frames_with_ids += 1
            track_ids.update(int(track_id) for track_id in boxes.id.tolist())
            if sample_track is None and len(boxes):
                sample_track = {
                    "xyxy": [round(float(value), 1) for value in boxes.xyxy[0]],
                    "track_id": int(boxes.id[0]),
                    "confidence": round(float(boxes.conf[0]), 3),
                    "class_id": int(boxes.cls[0]),
                }
        if frames_seen >= args.frames:
            break

    results.close()
    trackers = model.predictor.trackers
    tracker = trackers[0]
    print(f"frames: {frames_seen}")
    print(f"frames with track IDs: {frames_with_ids}")
    print(f"unique track IDs: {sorted(track_ids)}")
    print(f"tracker config: {model.predictor.args.tracker}")
    print(f"tracker class: {type(tracker).__module__}.{type(tracker).__name__}")
    print(f"tracker settings: {vars(tracker.args)}")
    print(f"sample track: {sample_track}")


if __name__ == "__main__":
    main()
