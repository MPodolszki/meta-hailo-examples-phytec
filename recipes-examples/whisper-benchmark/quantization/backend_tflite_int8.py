"""Whisper-tiny on the i.MX8MP, through a full-integer (int8) quantized TFLite
graph -- weights AND activations int8, model input/output tensors int8 too.

This is a sibling of TFLiteBackend, not a modification of it: the quantized
export exposes int8 tensors at every signature boundary (mel input, encoder
states, decoder logits), so every call has to affine-quantize/dequantize at
the Python edge using each tensor's own (scale, zero_point) -- and, critically,
the encoder's "output_0" and the decoder's "args_0" are NOT quantized with the
same (scale, zero_point) even though they carry the same logical tensor, so
the encoder output must be dequantized and re-quantized for the decoder
rather than passed through as raw int8 bytes.

NOT wired into the demo-whisper-benchmark package: see QUANTIZATION.md --
static PTQ makes this backend produce a garbage transcript on real hardware,
so it is kept here purely as a validated reference implementation of the
quantize/dequantize boundary, not as something to ship. To exercise it, drop
it into ../files/whisper_bench/ (it needs the package's relative imports) or
adjust the import below to match wherever you copy it.
"""

import os

import numpy as np

from whisper_bench.backend import Backend

VX_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"


def _quantize(value, scale, zero_point):
    q = np.round(value / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def _dequantize(value, scale, zero_point):
    return (value.astype(np.float32) - zero_point) * scale


class QuantizedTFLiteBackend(Backend):
    """Whisper-tiny, full-integer int8, optionally offloaded to the NPU."""

    def __init__(self, model_path, device="npu", num_threads=4,
                 cache_dir=None, chunk_length=10):
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:  # pragma: no cover - host-side development path
            from ai_edge_litert.interpreter import Interpreter, load_delegate

        self.device = device
        self.name = f"i.MX8MP {'NPU' if device == 'npu' else 'CPU'} (int8)"
        self.chunk_length = chunk_length
        self.model_path = str(model_path)

        delegates = []
        if device == "npu":
            if not os.path.exists(VX_DELEGATE_PATH):
                raise RuntimeError(
                    f"VX delegate not found at {VX_DELEGATE_PATH}; "
                    "install tensorflow-lite-vx-delegate"
                )
            self._enable_graph_cache(cache_dir)
            delegates.append(load_delegate(VX_DELEGATE_PATH))
        elif device != "cpu":
            raise ValueError(f"unsupported TFLite device: {device}")

        self._interpreter = Interpreter(
            model_path=self.model_path,
            experimental_delegates=delegates,
            num_threads=num_threads,
        )

        self._encode = self._interpreter.get_signature_runner("encode")
        self._decode = self._interpreter.get_signature_runner("decode")

        enc_in = self._encode.get_input_details()["args_0"]
        self._mel_shape = tuple(int(d) for d in enc_in["shape"])
        self._mel_q = enc_in["quantization"]

        enc_out = self._encode.get_output_details()["output_0"]
        self._enc_out_q = enc_out["quantization"]

        dec_in = self._decode.get_input_details()
        self.sequence_length = int(dec_in["args_1"]["shape"][1])
        self._dec_states_q = dec_in["args_0"]["quantization"]
        self._mask_q = dec_in["args_2"]["quantization"]

        dec_out = self._decode.get_output_details()["output_0"]
        self._logits_q = dec_out["quantization"]

        self._mask = _quantize(self._causal_mask(self.sequence_length),
                                *self._mask_q)

    @staticmethod
    def _enable_graph_cache(cache_dir):
        if not cache_dir:
            return
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("VIV_VX_ENABLE_CACHE_GRAPH_BINARY", "1")
        os.environ.setdefault("VIV_VX_CACHE_BINARY_GRAPH_DIR", str(cache_dir))

    @staticmethod
    def _causal_mask(seq_len, mask_value=-1e9):
        mask = np.triu(np.full((seq_len, seq_len), mask_value, dtype=np.float32), k=1)
        return mask.reshape(1, 1, seq_len, seq_len)

    def encode(self, mel):
        tensor = np.ascontiguousarray(mel[np.newaxis, ...], dtype=np.float32)
        if tensor.shape != self._mel_shape:
            raise ValueError(
                f"mel shape {tensor.shape} does not match the encoder "
                f"signature {self._mel_shape}"
            )
        quantized = _quantize(tensor, *self._mel_q)
        return self._encode(args_0=quantized)["output_0"]

    def decode_logits(self, encoded, token_ids, position):
        # encoded is int8 in the ENCODER's quantization; the decoder input has
        # its own (scale, zero_point) for the same logical tensor, so bounce
        # through float rather than passing the int8 bytes through unchanged.
        states = _dequantize(encoded, *self._enc_out_q)
        states = _quantize(states, *self._dec_states_q)

        tokens = np.ascontiguousarray(token_ids, dtype=np.int32)
        outputs = self._decode(args_0=states, args_1=tokens, args_2=self._mask)
        logits = outputs["output_0"][0, position]
        # Affine, monotonic, per-tensor quantization: argmax over raw int8
        # values is identical to argmax over the dequantized floats.
        return logits.astype(np.int32)

    def describe(self):
        return {
            "runtime": "TensorFlow Lite",
            "delegate": "VX (NPU)" if self.device == "npu" else "XNNPACK (CPU)",
            "model": os.path.basename(self.model_path),
            "sequence_length": self.sequence_length,
            "quantization": "full-integer int8 (static, weights+activations)",
        }
