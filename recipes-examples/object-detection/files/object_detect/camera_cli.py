"""Command line front-end: YOLOv8n object detection on a camera or video file.

There is deliberately no cv2.imshow() / GUI window here: this image does not
guarantee an X11/Wayland stack with GTK support in OpenCV is present. Instead,
each annotated frame overwrites --output, so it can be viewed without a local
display (scp/sshfs the file, or serve the directory it lives in).

object-detection-camera exposes every backend via -b/--backend, including
'compare' (both backends, side by side). object-detection-hailo-cam,
-hailo-video, -cpu-cam, -cpu-video, -compare-cam and -compare-video are
convenience wrappers around run() below that each fix one backend and one
source kind, the same way whisper-hailo/whisper-cpu wrap whisper_bench's
transcribe_cli.run().
"""

import argparse
import os
import subprocess
import sys
import time

import cv2

from .detector import BACKENDS, make_detector
from .postprocess import draw_detections, get_labels, preprocess

DEFAULT_DATA_DIR = "/usr/share/demo-object-detection"
DEFAULT_OUTPUT = "/tmp/object-detection-camera.jpg"

#: 'compare' isn't a detector backend (see detector.BACKENDS) -- it's a mode
#: of this CLI that runs every detector backend on the same frame.
CAMERA_BACKENDS = BACKENDS + ("compare",)

LABELS = {"hailo": "Hailo-8", "cpu": "i.MX8MP CPU"}


def _write_output(path, image):
    """Write atomically so a concurrent reader (e.g. a live viewer polling
    this file) never observes a half-written frame."""
    root, ext = os.path.splitext(path)
    tmp_path = f"{root}.tmp{ext}"
    cv2.imwrite(tmp_path, image)
    os.replace(tmp_path, path)


def _asset(data_dir, *parts):
    path = os.path.join(data_dir, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _video_source(value):
    """A camera index (e.g. "0") or a video file path."""
    try:
        return int(value)
    except ValueError:
        return value


#: The phyCAM-L/M sensors on this board (ar0144, ar0521, ...) are exposed to
#: the ISI capture node in this raw Bayer layout (V4L2 mbus code SGRBG8_1X8).
#: The ISI's own Bayer->RGB conversion produces all-black frames on this
#: board/kernel combination, so cameras are opened in raw Bayer instead and
#: debayered here in software, the same workaround used by
#: demo-celebrity-face-match (see its camera.py, CameraVM016).
_CSI1_DEVICE = "/dev/video-isi-csi1"
_CSI1_SIZE = "1280x800"


def _open_camera_capture(index):
    subprocess.run(
        f"setup-pipeline-csi1 -s {_CSI1_SIZE} -c {_CSI1_SIZE}",
        shell=True, check=True,
    )
    pipeline = (
        f"v4l2src device={_CSI1_DEVICE} ! "
        f"video/x-bayer,format=grbg,width=1280,height=800 ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"could not open camera index {index!r}", file=sys.stderr)
        return None
    return cap


def _open_capture(source):
    if isinstance(source, int):
        cap = _open_camera_capture(source)
        return (cap, True) if cap is not None else (None, False)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"could not open video file {source!r}", file=sys.stderr)
        return None, False
    return cap, False


def build_parser(prog="object-detection-camera", fixed_backend=None, require_source=False):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run YOLOv8n object detection on a live camera feed or a "
                    "video file. Each frame is annotated and written to "
                    "--output, overwriting the previous one.",
    )
    if fixed_backend is None:
        parser.add_argument(
            "-b", "--backend", choices=CAMERA_BACKENDS, default="hailo",
            help="inference backend, or 'compare' to run every backend side "
                 "by side (default: hailo)",
        )
    if require_source:
        parser.add_argument(
            "source", type=_video_source, metavar="SOURCE",
            help="camera index (e.g. 0) or a video file path",
        )
    else:
        parser.add_argument(
            "source", type=_video_source, metavar="SOURCE", nargs="?", default=0,
            help="camera index (e.g. 0) or a video file path (default: camera 0)",
        )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT, metavar="FILE",
        help=f"annotated JPEG frame, overwritten every frame (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-d", "--data-dir", default=DEFAULT_DATA_DIR, metavar="DIR",
        help=f"model and asset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "-s", "--score-thres", type=float, default=0.25, metavar="THRES",
        help="minimum detection confidence to draw (default: 0.25)",
    )
    parser.add_argument(
        "-n", "--frames", type=int, metavar="N",
        help="stop after N frames (default: run until Ctrl+C or end of stream)",
    )
    return parser


def _run_single(args, backend):
    try:
        labels_path = _asset(args.data_dir, "assets", "labels.txt")
        detector, extract = make_detector(backend, args.data_dir)
    except FileNotFoundError as error:
        print(f"missing demo data: {error}\n"
              f"install demo-object-detection-data or pass --data-dir",
              file=sys.stderr)
        return 2

    cap, needs_debayer = _open_capture(args.source)
    if cap is None:
        detector.close()
        return 2

    labels = get_labels(labels_path)
    model_h, model_w = detector.input_shape[:2]

    frame_count = 0
    start = time.perf_counter()
    try:
        while args.frames is None or frame_count < args.frames:
            ok, frame = cap.read()
            if not ok:
                if frame_count == 0:
                    print("could not read from source", file=sys.stderr)
                break
            if needs_debayer:
                frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2BGR)

            raw_output = detector.infer(preprocess(frame, model_w, model_h))
            detections = extract(raw_output, frame.shape,
                                  score_thres=args.score_thres)
            annotated = draw_detections(frame, detections, labels)
            _write_output(args.output, annotated)

            frame_count += 1
            fps = frame_count / (time.perf_counter() - start)
            names = ", ".join(labels[class_id] for _, class_id, _ in detections[:5])
            status = f"\rframe {frame_count:5d}  {fps:5.1f} fps  {len(detections)} detection(s)"
            if names:
                status += f": {names}"
            print(status, end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        cap.release()
        detector.close()

    return 0


def _label_frame(frame, text):
    origin = (10, 24)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _run_compare(args):
    detectors = {}
    try:
        labels_path = _asset(args.data_dir, "assets", "labels.txt")
        for backend in BACKENDS:
            detectors[backend] = make_detector(backend, args.data_dir)
    except FileNotFoundError as error:
        for detector, _ in detectors.values():
            detector.close()
        print(f"missing demo data: {error}\n"
              f"install demo-object-detection-data or pass --data-dir",
              file=sys.stderr)
        return 2

    cap, needs_debayer = _open_capture(args.source)
    if cap is None:
        for detector, _ in detectors.values():
            detector.close()
        return 2

    labels = get_labels(labels_path)
    frame_count = 0
    elapsed = {backend: 0.0 for backend in BACKENDS}
    try:
        while args.frames is None or frame_count < args.frames:
            ok, frame = cap.read()
            if not ok:
                if frame_count == 0:
                    print("could not read from source", file=sys.stderr)
                break
            if needs_debayer:
                frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2BGR)

            halves, status_parts = [], []
            for backend in BACKENDS:
                detector, extract = detectors[backend]
                model_h, model_w = detector.input_shape[:2]

                start = time.perf_counter()
                raw_output = detector.infer(preprocess(frame, model_w, model_h))
                detections = extract(raw_output, frame.shape,
                                      score_thres=args.score_thres)
                elapsed[backend] += time.perf_counter() - start

                fps = (frame_count + 1) / elapsed[backend]
                half = draw_detections(frame.copy(), detections, labels)
                halves.append(_label_frame(half, f"{LABELS[backend]}: {fps:.1f} fps"))
                status_parts.append(f"{LABELS[backend]} {fps:5.1f} fps")

            _write_output(args.output, cv2.hconcat(halves))

            frame_count += 1
            print(f"\rframe {frame_count:5d}  " + "  ".join(status_parts),
                  end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        cap.release()
        for detector, _ in detectors.values():
            detector.close()

    return 0


def main(argv=None, prog="object-detection-camera", fixed_backend=None,
         require_source=False):
    parser = build_parser(prog=prog, fixed_backend=fixed_backend,
                           require_source=require_source)
    args = parser.parse_args(argv)
    backend = fixed_backend or args.backend

    if backend == "compare":
        return _run_compare(args)
    return _run_single(args, backend)


def run(fixed_backend, require_source, argv=None):
    """Entry point shared by the fixed-backend convenience commands."""
    kind = "video" if require_source else "cam"
    prog = f"object-detection-{fixed_backend}-{kind}"
    return main(argv, prog=prog, fixed_backend=fixed_backend,
                require_source=require_source)


if __name__ == "__main__":
    sys.exit(main())
