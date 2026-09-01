"""YOLOv8n object detection on the i.MX8MP Cortex-A53 cores, via OpenCV DNN.

Runs the same uncompiled YOLOv8n ONNX graph Hailo Model Zoo compiled the
bundled HEF from (see hailo_model_zoo's cfg/networks/yolov8n.yaml), minus
Hailo's on-chip NMS postprocessing layer -- that graph's raw output is [1, 84,
8400]: 4 box coordinates plus 80 class scores per anchor, still needing decode
and NMS on the host, done by postprocess.extract_detections_cpu.
"""


class CPUDetector:
    """Runs the uncompiled YOLOv8n ONNX graph on the CPU via OpenCV's DNN module."""

    input_shape = (640, 640, 3)  # (height, width, channels)

    def __init__(self, onnx_path, num_threads=None):
        import cv2

        self.onnx_path = str(onnx_path)
        if num_threads:
            cv2.setNumThreads(num_threads)

        self._net = cv2.dnn.readNetFromONNX(self.onnx_path)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def infer(self, frame):
        """Run one preprocessed HxWxC uint8 frame through the model.

        Returns the raw [1, 84, 8400] output -- see postprocess.extract_detections_cpu.
        """
        import cv2

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, swapRB=True)
        self._net.setInput(blob)
        return self._net.forward()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
