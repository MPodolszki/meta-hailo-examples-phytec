"""Command line front-end: Hailo-8 vs i.MX8MP CPU object detection benchmark."""

import argparse
import os
import sys
import time

import cv2

from .detector import BACKENDS, make_detector
from .postprocess import get_labels, preprocess

DEFAULT_DATA_DIR = "/usr/share/demo-object-detection"

LABELS = {"hailo": "Hailo-8", "cpu": "i.MX8MP CPU"}


def _asset(data_dir, *parts):
    path = os.path.join(data_dir, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        prog="object-detection-benchmark",
        description="Compare YOLOv8n inference throughput on the Hailo-8 AI "
                    "accelerator and the i.MX8MP Cortex-A53 cores, on the "
                    "same image.",
    )
    parser.add_argument(
        "-b", "--backend", action="append", choices=BACKENDS,
        metavar="{hailo,cpu}",
        help="backend to measure; repeatable (default: both)",
    )
    parser.add_argument(
        "-i", "--image", metavar="FILE",
        help="image to run detection on (default: the bundled sample)",
    )
    parser.add_argument(
        "-d", "--data-dir", default=DEFAULT_DATA_DIR, metavar="DIR",
        help=f"model and asset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "-n", "--repeats", type=int, default=20, metavar="N",
        help="inferences to time per backend, after a warm-up run (default: 20)",
    )
    parser.add_argument(
        "-s", "--score-thres", type=float, default=0.25, metavar="THRES",
        help="minimum detection confidence to report (default: 0.25)",
    )
    return parser


def _run(backend, image, data_dir, repeats, score_thres):
    """Time *repeats* inferences of *backend* on *image*.

    Returns (load_seconds, avg_seconds_per_inference, detections).
    """
    load_start = time.perf_counter()
    detector, extract = make_detector(backend, data_dir)
    load_seconds = time.perf_counter() - load_start

    try:
        model_h, model_w = detector.input_shape[:2]
        frame = preprocess(image, model_w, model_h)

        detector.infer(frame)  # warm-up: excludes one-time setup costs

        start = time.perf_counter()
        for _ in range(repeats):
            raw_output = detector.infer(frame)
        elapsed = time.perf_counter() - start
    finally:
        detector.close()

    detections = extract(raw_output, image.shape, score_thres=score_thres)
    return load_seconds, elapsed / repeats, detections


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    backends = args.backend or list(BACKENDS)

    try:
        labels_path = _asset(args.data_dir, "assets", "labels.txt")
        image_path = args.image or _asset(args.data_dir, "assets", "dog_bicycle.jpg")
    except FileNotFoundError as error:
        print(f"missing demo data: {error}\n"
              f"install demo-object-detection-data or pass --data-dir/--image",
              file=sys.stderr)
        return 2

    image = cv2.imread(image_path)
    if image is None:
        print(f"could not read image: {image_path}", file=sys.stderr)
        return 2
    labels = get_labels(labels_path)

    print(f"image: {os.path.basename(image_path)} "
          f"({image.shape[1]}x{image.shape[0]}), {args.repeats} inferences/backend\n")

    results = {}
    for backend in backends:
        label = LABELS[backend]
        print(f"[{label}] loading + running...", file=sys.stderr)
        try:
            results[backend] = _run(
                backend, image, args.data_dir, args.repeats, args.score_thres)
        except Exception as error:  # noqa: BLE001 - report and keep going
            print(f"[{label}] unavailable: {error}", file=sys.stderr)
            results[backend] = None

    name_w = max(len(LABELS[b]) for b in backends)
    print(f"{'backend':<{name_w}}  {'load':>7}  {'latency':>9}  {'fps':>8}  detections")
    for backend in backends:
        label = LABELS[backend]
        result = results[backend]
        if result is None:
            print(f"{label:<{name_w}}  {'--':>7}  {'--':>9}  {'--':>8}  unavailable")
            continue
        load_seconds, avg_seconds, detections = result
        names = ", ".join(labels[cid] for _, cid, _ in detections[:5])
        summary = f"{len(detections)} ({names})" if names else str(len(detections))
        print(f"{label:<{name_w}}  {load_seconds:6.2f}s  {1000 * avg_seconds:7.1f}ms  "
              f"{1 / avg_seconds:8.2f}  {summary}")

    if results.get("hailo") and results.get("cpu"):
        speedup = results["cpu"][1] / results["hailo"][1]
        print(f"\nHailo-8 is {speedup:.1f}x faster than the CPU here.")

    return 0 if any(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
