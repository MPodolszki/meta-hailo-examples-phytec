"""Backend selection: Hailo-8 AI accelerator or i.MX8MP CPU.

Both backends run YOLOv8n over the same 640x640 letterboxed input (see
postprocess.preprocess) and are tuned to the same NMS thresholds, so their
detections on the same image are directly comparable.
"""

import os

from .postprocess import extract_detections, extract_detections_cpu

BACKENDS = ("hailo", "cpu")


def _asset(data_dir, *parts):
    path = os.path.join(data_dir, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def make_detector(backend, data_dir):
    """Build the detector for *backend* and its matching extraction function.

    Returns (detector, extract_fn). detector exposes .input_shape,
    .infer(frame) and .close(); extract_fn(raw_output, image_shape,
    score_thres) turns that backend's raw output into the same
    (score, class_id, (x_min, y_min, x_max, y_max)) list either way.

    Raises FileNotFoundError if that backend's model file is missing.
    """
    if backend == "hailo":
        from .hailo_infer import HailoDetector

        hef_path = _asset(data_dir, "hailo", "yolov8n.hef")
        return HailoDetector(hef_path), extract_detections

    from .cpu_infer import CPUDetector

    onnx_path = _asset(data_dir, "cpu", "yolov8n.onnx")
    return CPUDetector(onnx_path), extract_detections_cpu
