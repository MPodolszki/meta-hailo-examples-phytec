"""Pre/post-processing around the bundled YOLOv8n HEF's on-chip NMS output."""

import cv2
import numpy as np


def get_labels(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def preprocess(image, model_w, model_h):
    """Letterbox-resize `image` to model_w x model_h, keeping aspect ratio.

    Padding (grey, 114) is only ever added along one axis -- extract_detections
    below removes exactly that padding again when mapping boxes back to the
    original image.
    """
    img_h, img_w = image.shape[:2]
    scale = min(model_w / img_w, model_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    padded = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
    x_off, y_off = (model_w - new_w) // 2, (model_h - new_h) // 2
    padded[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return padded


def extract_detections(raw_output, image_shape, score_thres=0.25, max_boxes=100):
    """Flatten HailoRT's per-class NMS output into image-space detections.

    Args:
        raw_output: list with one [N, 5] array per class, boxes as
            (y_min, x_min, y_max, x_max, score) normalized to the letterboxed
            square (see `preprocess`).
        image_shape: shape of the original (pre-letterbox) image.
        score_thres: minimum score to keep a detection.
        max_boxes: cap on the number of detections returned, highest score first.

    Returns:
        list of (score, class_id, (x_min, y_min, x_max, y_max)) tuples, sorted
        by descending score, with box coordinates in original-image pixels.
    """
    img_h, img_w = image_shape[:2]
    size = max(img_h, img_w)
    pad = abs(img_h - img_w) // 2

    detections = []
    for class_id, class_boxes in enumerate(raw_output):
        for det in class_boxes:
            score = float(det[4])
            if score < score_thres:
                continue
            y_min, x_min, y_max, x_max = (float(v) * size for v in det[:4])
            if img_h != size:
                y_min -= pad
                y_max -= pad
            if img_w != size:
                x_min -= pad
                x_max -= pad
            detections.append(
                (score, class_id, (int(x_min), int(y_min), int(x_max), int(y_max)))
            )

    detections.sort(reverse=True, key=lambda d: d[0])
    return detections[:max_boxes]


def extract_detections_cpu(raw_output, image_shape, score_thres=0.25, iou_thres=0.7,
                            max_boxes=100, model_size=640):
    """Decode + NMS a raw YOLOv8 output into detections.

    Args:
        raw_output: the CPUDetector's [1, 84, 8400] output -- 4 box
            coordinates (cx, cy, w, h, in model pixel space) plus 80 class
            scores, per anchor. Unlike the Hailo HEF, this graph has no
            postprocessing baked in, so decoding (highest-scoring class per
            anchor) and NMS both happen here, via cv2.dnn.NMSBoxes.
        image_shape: shape of the original (pre-letterbox) image.
        score_thres: minimum score to keep a detection.
        iou_thres: NMS IoU threshold -- matches the Hailo HEF's compiled-in
            NMS config so both backends are tuned the same way.
        max_boxes: cap on the number of detections returned, highest score first.
        model_size: the square input size detection boxes are decoded against.

    Returns:
        list of (score, class_id, (x_min, y_min, x_max, y_max)) tuples, sorted
        by descending score, with box coordinates in original-image pixels --
        the same format as extract_detections.
    """
    preds = raw_output[0]  # [84, 8400]
    boxes_cxcywh = preds[:4].T  # [8400, 4], model pixel space
    class_scores = preds[4:].T  # [8400, 80]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    keep = scores >= score_thres
    boxes_cxcywh, class_ids, scores = boxes_cxcywh[keep], class_ids[keep], scores[keep]
    if len(scores) == 0:
        return []

    cx, cy, w, h = boxes_cxcywh.T
    boxes_xywh = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)

    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), score_thres, iou_thres)
    if len(indices) == 0:
        return []
    indices = np.asarray(indices).flatten()

    img_h, img_w = image_shape[:2]
    size = max(img_h, img_w)
    pad = abs(img_h - img_w) // 2
    scale = size / model_size

    detections = []
    for i in indices:
        x, y, w, h = boxes_xywh[i]
        x_min, y_min = x * scale, y * scale
        x_max, y_max = (x + w) * scale, (y + h) * scale
        if img_h != size:
            y_min -= pad
            y_max -= pad
        if img_w != size:
            x_min -= pad
            x_max -= pad
        detections.append(
            (float(scores[i]), int(class_ids[i]),
             (int(x_min), int(y_min), int(x_max), int(y_max)))
        )

    detections.sort(reverse=True, key=lambda d: d[0])
    return detections[:max_boxes]


def id_to_color(class_id):
    rng = np.random.RandomState(class_id)
    return tuple(int(c) for c in rng.randint(0, 255, size=3))


def draw_detections(image, detections, labels):
    for score, class_id, (x_min, y_min, x_max, y_max) in detections:
        color = id_to_color(class_id)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        text = f"{labels[class_id]}: {score * 100:.0f}%"
        origin = (x_min + 4, y_min + 18)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return image
