"""Whisper-tiny on the Hailo-8 accelerator, via the HailoRT InferModel API.

Encoder and decoder are separate HEFs, both compiled for a 10 s window:

    encoder: [1, 1, 1000, 80] mel (NHWC) -> [1, 1, 500, 384] encoder states
    decoder: encoder states + token embeddings
                                         -> per-layer logit slices, seq len 32

The Hailo-8 decoder HEF is compiled without the token embedding lookup, so
Gather + Add(positional) + Transpose run on the CPU here. That is a property
of the compiled model, not a shortcut taken by this demo — the equivalent
TFLite graph keeps EMBEDDING_LOOKUP inside the model.
"""

import os

import numpy as np

from .backend import Backend


class HailoBackend(Backend):
    """Whisper-tiny executed on a Hailo-8 over PCIe."""

    name = "Hailo-8"
    device = "hailo8"

    def __init__(self, encoder_path, decoder_path, embedding_path,
                 positional_path, chunk_length=10, timeout_ms=10_000):
        from hailo_platform import (HEF, FormatType, HailoSchedulingAlgorithm,
                                    VDevice)

        self.encoder_path = str(encoder_path)
        self.decoder_path = str(decoder_path)
        self.chunk_length = chunk_length
        self._timeout_ms = timeout_ms

        # Host-side half of the decoder's tokenization stage.
        self._token_embedding = np.load(str(embedding_path))
        self._positional = np.load(str(positional_path))

        encoder_hef = HEF(self.encoder_path)
        encoder_shape = encoder_hef.get_input_vstream_infos()[0].shape
        # VStream shapes are (height, width, features); width is the frame axis.
        self._encoder_frames = int(encoder_shape[1])
        if self._encoder_frames != chunk_length * 100:
            raise ValueError(
                f"encoder HEF expects {self._encoder_frames / 100:g} s windows, "
                f"demo is configured for {chunk_length} s"
            )

        decoder_hef = HEF(self.decoder_path)
        self._decoder_name = decoder_hef.get_network_group_names()[0]
        self._output_names = decoder_hef.get_sorted_output_names()
        # The decoder's vocabulary is split across several output layers that
        # have to be concatenated back together along the vocabulary axis.
        self._logit_outputs = [n for n in self._output_names if "conv" in n]
        self.sequence_length = int(
            decoder_hef.get_output_vstream_infos()[0].shape[1]
        )

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"

        self._vdevice = VDevice(params)
        self._stack = []

        # Anything that fails from here on must still hand the device back,
        # otherwise a failed Hailo run would block the accelerator for the rest
        # of the benchmark.
        try:
            encoder_model = self._vdevice.create_infer_model(self.encoder_path)
            encoder_model.input().set_format_type(FormatType.FLOAT32)
            encoder_model.output().set_format_type(FormatType.FLOAT32)

            decoder_model = self._vdevice.create_infer_model(self.decoder_path)
            decoder_model.input(f"{self._decoder_name}/input_layer1").set_format_type(
                FormatType.FLOAT32)
            decoder_model.input(f"{self._decoder_name}/input_layer2").set_format_type(
                FormatType.FLOAT32)
            for name in self._output_names:
                decoder_model.output(name).set_format_type(FormatType.FLOAT32)

            self._encoder_model = encoder_model
            self._decoder_model = decoder_model
            self._encoder = self._enter(encoder_model.configure())
            self._decoder = self._enter(decoder_model.configure())
            self._encoder_bindings = self._encoder.create_bindings()
            self._decoder_bindings = self._decoder.create_bindings()
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

    # -- inference --------------------------------------------------------

    def encode(self, mel):
        # [n_mels, frames] -> NHWC [1, 1, frames, n_mels]
        tensor = np.ascontiguousarray(
            mel.T[np.newaxis, np.newaxis, ...], dtype=np.float32
        )

        self._encoder_bindings.input().set_buffer(tensor)
        self._encoder_bindings.output().set_buffer(
            np.zeros(self._encoder_model.output().shape, dtype=np.float32)
        )
        self._encoder.run([self._encoder_bindings], self._timeout_ms)
        return self._encoder_bindings.output().get_buffer()

    def _embed_tokens(self, token_ids):
        """Gather + positional Add + Transpose, the part left out of the HEF."""
        gathered = self._token_embedding[token_ids] + self._positional
        return np.transpose(np.expand_dims(gathered, axis=0), (0, 2, 1, 3))

    def decode_logits(self, encoded, token_ids, position):
        embedded = np.ascontiguousarray(self._embed_tokens(token_ids), dtype=np.float32)

        bindings = self._decoder_bindings
        bindings.input(f"{self._decoder_name}/input_layer1").set_buffer(encoded)
        bindings.input(f"{self._decoder_name}/input_layer2").set_buffer(embedded)
        for name in self._output_names:
            bindings.output(name).set_buffer(
                np.zeros(self._decoder_model.output(name).shape, dtype=np.float32)
            )

        self._decoder.run([bindings], self._timeout_ms)

        logits = np.concatenate(
            [bindings.output(name).get_buffer() for name in self._logit_outputs],
            axis=2,
        )
        return logits[0, position]

    def describe(self):
        return {
            "runtime": "HailoRT",
            "delegate": "Hailo-8 NN core",
            "model": "{} + {}".format(
                os.path.basename(self.encoder_path),
                os.path.basename(self.decoder_path),
            ),
            "sequence_length": self.sequence_length,
        }
