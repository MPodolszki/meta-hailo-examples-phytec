"""YOLOv8n object detection on the Hailo-8 accelerator, via the HailoRT InferModel API.

The bundled HEF (Hailo Model Zoo yolov8n, compiled for Hailo-8) has HailoRT's
on-chip NMS postprocessing built in, so the single output tensor is already a
per-class list of decoded (y_min, x_min, y_max, x_max, score) boxes rather than
raw model logits -- see object_detect/postprocess.py.
"""


class HailoDetector:
    """Runs a single-input, single-output detection HEF on a Hailo-8."""

    def __init__(self, hef_path, timeout_ms=10_000):
        from hailo_platform import HEF, HailoSchedulingAlgorithm, VDevice

        self.hef_path = str(hef_path)
        self._timeout_ms = timeout_ms

        hef = HEF(self.hef_path)
        input_info = hef.get_input_vstream_infos()[0]
        self.input_name = input_info.name
        self.input_shape = tuple(input_info.shape)  # (height, width, channels)

        output_names = [info.name for info in hef.get_output_vstream_infos()]

        params = VDevice.create_params()
        # Required for the InferModel API to activate the network group's
        # streams automatically; without it, run() fails with
        # HAILO_STREAM_NOT_ACTIVATED.
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)
        self._stack = []

        # Anything that fails from here on must still hand the device back,
        # otherwise a failed Hailo run would block the accelerator afterwards.
        try:
            self._model = self._vdevice.create_infer_model(self.hef_path)

            # The HEF's vstream infos describe the raw tensor shape, but a
            # HailoRT NMS-by-class output packs an extra per-class box count
            # into the frame, so the buffer must be sized from the configured
            # InferModel's output (which accounts for that) rather than from
            # hef.get_output_vstream_infos().
            self._output_names = output_names
            self._output_shapes = {
                name: tuple(self._model.output(name).shape) for name in output_names
            }
            self._output_dtypes = {
                name: str(self._model.output(name).format.type).split(".")[-1].lower()
                for name in output_names
            }

            self._configured = self._enter(self._model.configure())
            self._bindings = self._configured.create_bindings()
        except BaseException:
            self.close()
            raise

    def _enter(self, context):
        self._stack.append(context)
        return context.__enter__()

    def close(self):
        while self._stack:
            try:
                self._stack.pop().__exit__(None, None, None)
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        vdevice = getattr(self, "_vdevice", None)
        if vdevice is not None:
            vdevice.release()
            self._vdevice = None

    def infer(self, frame):
        """Run one preprocessed HxWxC frame through the model.

        Returns the raw HailoRT output: a list with one entry per class, each
        a [N, 5] array of (y_min, x_min, y_max, x_max, score) boxes normalized
        to the padded square the frame was letterboxed into.
        """
        import numpy as np

        self._bindings.input(self.input_name).set_buffer(np.ascontiguousarray(frame))
        for name in self._output_names:
            self._bindings.output(name).set_buffer(
                np.empty(self._output_shapes[name],
                          dtype=getattr(np, self._output_dtypes[name]))
            )

        self._configured.run([self._bindings], self._timeout_ms)
        return self._bindings.output(self._output_names[0]).get_buffer()
