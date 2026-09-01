"""Whisper-tiny on the i.MX8MP Cortex-A53 cores, through TensorFlow Lite.

One TFLite file carries both an ``encode`` and a ``decode`` signature:

    encode: [1, 80, 1000] mel        -> [1, 500, 384] encoder states
    decode: encoder states,
            [1, 128] int32 token ids,
            [1, 1, 128, 128] additive causal mask
                                     -> [1, 128, 51865] logits

Runs on the CPU via XNNPACK only. There used to be an NPU (VeriSilicon
VIP8000, via NXP's VX delegate) code path here, removed after measuring it:
the NPU was 5.3x *slower* than the CPU on this model (see README.md) --
VIP8000 is a quantized-CNN engine, and Whisper's BATCH_MATMUL/TRANSPOSE/
layernorm chains don't map onto it. NXP's own eIQ model zoo speech-recognition
example for this NPU uses a CNN (wav2letter), not a transformer, for the same
reason. Keeping a working-but-misleading `-b npu` option around would suggest
the i.MX8MP NPU is a viable path for this workload, which it is not.
"""

import os

import numpy as np

from .backend import Backend

MASK_VALUE = -1e9


class TFLiteBackend(Backend):
    """Whisper-tiny executed by TFLite on the Cortex-A53 cores (XNNPACK)."""

    def __init__(self, model_path, num_threads=4, chunk_length=10):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:  # pragma: no cover - host-side development path
            from ai_edge_litert.interpreter import Interpreter

        self.device = "cpu"
        self.name = "i.MX8MP CPU"
        self.chunk_length = chunk_length
        self.model_path = str(model_path)

        self._interpreter = Interpreter(
            model_path=self.model_path,
            num_threads=num_threads,
        )

        self._encode = self._interpreter.get_signature_runner("encode")
        self._decode = self._interpreter.get_signature_runner("decode")

        decode_inputs = self._decode.get_input_details()
        self.sequence_length = int(decode_inputs["args_1"]["shape"][1])
        self._token_dtype = decode_inputs["args_1"]["dtype"]

        encode_inputs = self._encode.get_input_details()
        self._mel_shape = tuple(int(dim) for dim in encode_inputs["args_0"]["shape"])

        self._mask = self._causal_mask(self.sequence_length)

    @staticmethod
    def _causal_mask(seq_len):
        mask = np.triu(np.full((seq_len, seq_len), MASK_VALUE, dtype=np.float32), k=1)
        return mask.reshape(1, 1, seq_len, seq_len)

    def encode(self, mel):
        # [n_mels, frames] -> [1, n_mels, frames]
        tensor = np.ascontiguousarray(mel[np.newaxis, ...], dtype=np.float32)
        if tensor.shape != self._mel_shape:
            raise ValueError(
                f"mel shape {tensor.shape} does not match the encoder "
                f"signature {self._mel_shape}"
            )
        return self._encode(args_0=tensor)["output_0"]

    def decode_logits(self, encoded, token_ids, position):
        tokens = np.ascontiguousarray(token_ids, dtype=self._token_dtype)
        outputs = self._decode(args_0=encoded, args_1=tokens, args_2=self._mask)
        return outputs["output_0"][0, position]

    def describe(self):
        return {
            "runtime": "TensorFlow Lite",
            "delegate": "XNNPACK (CPU)",
            "model": os.path.basename(self.model_path),
            "sequence_length": self.sequence_length,
        }
