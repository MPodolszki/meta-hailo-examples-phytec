#!/usr/bin/env python3
"""Post-training full-integer (int8) quantization of whisper_tiny_10s_f32.tflite.

Runs on an x86 dev host, NOT on the board. Produces whisper_tiny_10s_i8.tflite
next to the input model, plus a text report of the op set (to confirm no
STABLEHLO ops crept in -- TFLite 2.15 on the target rejects those, see
tflite-215-no-stablehlo).

Steps this script performs (see QUANTIZATION.md for the full narrative):
  1. Load the float32 model with ai_edge_litert.
  2. Build a calibration set for BOTH signatures ("encode" and "decode") by
     actually running the float model end-to-end (encode + greedy decode) on
     a handful of real-audio windows, capturing every intermediate decoder
     input along the way. Calibrating "decode" with only forced-prompt inputs
     would under-represent the activation range once free generation departs
     from the prompt, so real generated token sequences are used instead.
  3. Hand both signatures' calibration data to ai_edge_quantizer with the
     static_wi8_ai8 recipe (both weights AND activations quantized to int8 --
     the point of this exercise is to test whether that lets the i.MX8MP NPU
     execute Whisper on its native int8 CNN path instead of the fp16 path).
  4. Export the quantized model and dump its op set for the STABLEHLO check.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Reuse the recipe's own whisper_bench package (pure numpy/stdlib, no board
# dependency) instead of duplicating audio.py/tokenizer.py here.
RECIPE_FILES = Path(__file__).resolve().parent.parent / "files"
sys.path.insert(0, str(RECIPE_FILES))

from whisper_bench import audio as audio_module  # noqa: E402
from whisper_bench.tokenizer import (  # noqa: E402
    FORCED_DECODER_IDS,
    WhisperTokenizer,
    clean_transcription,
)

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
CALIB_DIR = HERE / "calib_wav"
FLOAT_MODEL = MODEL_DIR / "whisper_tiny_10s_f32.tflite"
INT8_MODEL = MODEL_DIR / "whisper_tiny_10s_i8.tflite"
MEL_FILTERS = MODEL_DIR / "mel_filters.npz"
VOCAB = MODEL_DIR / "vocab.json"

SEQ_LEN = 128
MASK_VALUE = -1e9


def causal_mask(seq_len=SEQ_LEN):
    mask = np.triu(np.full((seq_len, seq_len), MASK_VALUE, dtype=np.float32), k=1)
    return mask.reshape(1, 1, seq_len, seq_len)


def build_windows(filters):
    """Real-audio calibration windows: a few different 10 s crops of jfk.wav.

    A single 11 s recording cannot give the calibration set true acoustic
    diversity, but shifting the window changes which words/silence fall
    where in the fixed 10 s frame, which is the axis that actually matters
    for this model (position-dependent decoder activations). This is a
    documented limitation, not an attempt to claim a representative corpus.
    """
    raw = audio_module.load_wav(CALIB_DIR / "jfk.wav")
    raw = audio_module.normalize_peak(raw)
    n = audio_module.N_SAMPLES  # 10 s at 16 kHz

    windows = []
    offsets_s = [0.0, 0.4, 1.0, 1.6]  # 0.4s matches the silence-trim point
    for offset_s in offsets_s:
        start = int(offset_s * audio_module.SAMPLE_RATE)
        window = audio_module.pad_or_trim(raw[start:start + n], n)
        mel = audio_module.log_mel_spectrogram(window, filters)
        windows.append(mel.astype(np.float32))
    return windows


def run_float_model_and_collect(interpreter, mel_windows):
    """Encode + greedy-decode each window on the float model.

    Returns (encode_calib, decode_calib, transcripts) where the calib lists
    are ready to hand to ai_edge_quantizer.
    """
    encode_runner = interpreter.get_signature_runner("encode")
    decode_runner = interpreter.get_signature_runner("decode")
    mask = causal_mask()

    encode_calib = []
    decode_calib = []
    transcripts = []

    for mel in mel_windows:
        tensor = mel[np.newaxis, ...].astype(np.float32)
        encode_calib.append({"args_0": tensor})

        encoded = encode_runner(args_0=tensor)["output_0"]

        token_ids = np.zeros((1, SEQ_LEN), dtype=np.int32)
        for index, token in enumerate(FORCED_DECODER_IDS):
            token_ids[0][index] = token

        first = len(FORCED_DECODER_IDS) - 1
        generated = []
        seen_ngrams = set()

        for position in range(first, SEQ_LEN - 1):
            decode_calib.append({
                "args_0": encoded.copy(),
                "args_1": token_ids.copy(),
                "args_2": mask,
            })
            logits = decode_runner(
                args_0=encoded, args_1=token_ids, args_2=mask,
            )["output_0"][0, position]
            next_token = int(np.argmax(logits))

            if next_token == 50257:  # EOT
                break
            generated.append(next_token)
            token_ids[0][position + 1] = next_token

            if len(generated) >= 4:
                ngram = tuple(generated[-4:])
                if ngram in seen_ngrams:
                    del generated[-4:]
                    break
                seen_ngrams.add(ngram)

        transcripts.append(generated)

    return encode_calib, decode_calib, transcripts


def dump_opcodes(model_path):
    """Return the set of custom op names in a .tflite flatbuffer.

    STABLEHLO_COMPOSITE ops (see tflite-215-no-stablehlo) always show up as
    CUSTOM ops, never as a builtin code, so checking the custom op codes is
    sufficient to rule them out.
    """
    import tflite  # pip package "tflite", pure flatbuffer reader

    with open(model_path, "rb") as handle:
        buf = bytearray(handle.read())
    model = tflite.Model.GetRootAsModel(buf, 0)

    names = set()
    for i in range(model.OperatorCodesLength()):
        code = model.OperatorCodes(i)
        custom = code.CustomCode()
        if custom:
            names.add(f"CUSTOM:{custom.decode()}")
    return names


def main():
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import quantizer, recipe

    filters = audio_module.load_mel_filters(MEL_FILTERS)
    tokenizer = WhisperTokenizer(VOCAB)

    print("[1/5] loading float model + building calibration windows")
    interp = Interpreter(model_path=str(FLOAT_MODEL))
    windows = build_windows(filters)

    print(f"[2/5] running float model on {len(windows)} windows to collect "
          f"calibration activations (encode + full greedy decode each)")
    encode_calib, decode_calib, transcripts = run_float_model_and_collect(
        interp, windows)
    print(f"      {len(encode_calib)} encode samples, "
          f"{len(decode_calib)} decode samples")
    for tokens in transcripts:
        print("      float transcript:", clean_transcription(tokenizer.decode(tokens)))

    print("[3/5] calibrating + quantizing (static int8, weights+activations)")
    qt = quantizer.Quantizer(str(FLOAT_MODEL))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    calibration_data = {"encode": encode_calib, "decode": decode_calib}
    calib_result = qt.calibrate(calibration_data)
    result = qt.quantize(calib_result)
    result.export_model(str(INT8_MODEL))
    print(f"      wrote {INT8_MODEL} "
          f"({INT8_MODEL.stat().st_size / 1e6:.1f} MB, was "
          f"{FLOAT_MODEL.stat().st_size / 1e6:.1f} MB)")

    print("[4/5] checking the quantized model's op set for STABLEHLO ops")
    ops = dump_opcodes(INT8_MODEL)
    stablehlo = [op for op in ops if "STABLEHLO" in op or "stablehlo" in op]
    if stablehlo:
        print(f"      FOUND STABLEHLO OPS: {stablehlo} -- this model will "
              f"NOT load on the target's TFLite 2.15")
    else:
        print(f"      clean: {sorted(ops)}")

    print("[5/5] done -- validate with backend_tflite_int8.QuantizedTFLiteBackend, "
          "the exported model's I/O tensors are int8, not float32")


if __name__ == "__main__":
    main()
