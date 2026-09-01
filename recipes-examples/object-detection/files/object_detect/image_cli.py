"""Command line front-end: YOLOv8n object detection on a single image."""

import argparse
import os
import sys

import cv2

from .detector import BACKENDS, make_detector
from .postprocess import draw_detections, get_labels, preprocess

DEFAULT_DATA_DIR = "/usr/share/demo-object-detection"


def _asset(data_dir, *parts):
    path = os.path.join(data_dir, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        prog="object-detection-image",
        description="Run YOLOv8n object detection on a single image, on "
                    "either the Hailo-8 AI accelerator or the i.MX8MP CPU.",
    )
    parser.add_argument(
        "-b", "--backend", choices=BACKENDS, default="hailo",
        help="inference backend (default: hailo)",
    )
    parser.add_argument(
        "-i", "--image", metavar="FILE",
        help="image to run detection on (default: the bundled sample)",
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE",
        help="where to save the annotated image "
             "(default: <image>_detected.jpg next to the input)",
    )
    parser.add_argument(
        "-d", "--data-dir", default=DEFAULT_DATA_DIR, metavar="DIR",
        help=f"model and asset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "-s", "--score-thres", type=float, default=0.25, metavar="THRES",
        help="minimum detection confidence to draw (default: 0.25)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        labels_path = _asset(args.data_dir, "assets", "labels.txt")
        image_path = args.image or _asset(args.data_dir, "assets", "dog_bicycle.jpg")
        detector, extract = make_detector(args.backend, args.data_dir)
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
    try:
        model_h, model_w = detector.input_shape[:2]
        raw_output = detector.infer(preprocess(image, model_w, model_h))
    finally:
        detector.close()

    detections = extract(raw_output, image.shape, score_thres=args.score_thres)
    annotated = draw_detections(image, detections, labels)

    output_path = args.output or f"{os.path.splitext(image_path)[0]}_detected.jpg"
    cv2.imwrite(output_path, annotated)

    print(f"[{args.backend}] {len(detections)} detection(s) in {os.path.basename(image_path)}:")
    for score, class_id, box in detections:
        print(f"  {labels[class_id]:<15} {score * 100:5.1f}%  box={box}")
    print(f"saved annotated image to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
