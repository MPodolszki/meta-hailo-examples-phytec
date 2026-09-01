# Post-training int8 quantization of whisper_tiny_10s_f32.tflite

Goal: quantize the TFLite Whisper-tiny graph used by the `npu`/`cpu` backends
to int8 (weights *and* activations), to test whether that lets the i.MX8MP
NPU (VeriSilicon VIP8000, a quantized-CNN engine) execute more of the model on
its native int8 path instead of falling back to fp16/CPU as documented in
README.md ("Why the NPU loses to its own CPU").

## Tooling

`ai-edge-quantizer` (Google's post-training quantizer for LiteRT/TFLite
models), because it operates directly on an *already-exported* `.tflite`
flatbuffer. We only have the compiled graph from `leuconoe/whisper-tiny-litert`
(see `demo-whisper-benchmark-data_1.0.bb`), not the original PyTorch/TF source
model, so a converter that requires a SavedModel/Keras model (the standard
`tf.lite.TFLiteConverter` quantization path) is not an option here.

Everything below runs on an x86 dev host, never on the board.

## Steps taken

1. **Install tooling** (host, `--user --break-system-packages` since the
   system Python is Debian-managed and has no working `venv`):
   ```sh
   pip install --user --break-system-packages \
       ai-edge-quantizer ai-edge-litert tensorflow-cpu numpy tflite
   ```

2. **Fetch the exact float source model** the recipe already pins, and verify
   its checksum against `demo-whisper-benchmark-data_1.0.bb`:
   ```sh
   curl -sL -o whisper_tiny_10s_f32.tflite \
     https://huggingface.co/leuconoe/whisper-tiny-litert/resolve/89cb4079401993912e02371838feca156c9e013e/whisper_tiny_10s_f32.tflite
   sha256sum whisper_tiny_10s_f32.tflite
   # 823c11176e8ea1d3ad1c9da2512a6e42431f800780cd1c3df92ba8aa9b11ebbe -- matches
   ```
   Also fetched `mel_filters.npz` and `vocab.json` (same URLs as the recipe)
   and the bundled `jfk.wav` sample, needed to build calibration data and to
   sanity-check the result.

3. **Build a calibration set for BOTH TFLite signatures.** The model exposes
   `encode` (mel -> encoder states) and `decode` (encoder states + token ids +
   causal mask -> logits); ai-edge-quantizer calibrates each signature
   separately, so both need real activation samples:
   - `encode` calibration inputs: the mel spectrogram of four 10 s crops of
     `jfk.wav` at different offsets (0.0 s, 0.4 s, 1.0 s, 1.6 s). A single
     11 s recording has no real acoustic diversity, but shifting the window
     changes which words/silence land in which position of the fixed 10 s
     frame, which is the axis that matters for a fixed-shape, no-KV-cache
     decoder. **This is a small, single-speaker calibration set** -- a real
     deployment would want many speakers/languages/noise conditions.
   - `decode` calibration inputs: rather than only the forced 4-token prompt,
     the float model's own greedy decode loop was run to completion for each
     of the four windows, and *every* decoder call along the way (each
     generated token's encoder-states/token-ids/mask triple) was recorded.
     This gave 103 decode calibration samples covering the full range of
     decoder positions actually exercised, not just the start of the prompt.
   - Script: `quantize_int8.py`, step `[2/5]`.

4. **Calibrate + quantize** with `ai_edge_quantizer`:
   ```python
   from ai_edge_quantizer import quantizer, recipe
   qt = quantizer.Quantizer("whisper_tiny_10s_f32.tflite")
   qt.load_quantization_recipe(recipe.static_wi8_ai8())   # int8 weights + int8 activations
   calib_result = qt.calibrate({"encode": encode_calib, "decode": decode_calib})
   qt.quantize(calib_result).export_model("whisper_tiny_10s_i8.tflite")
   ```
   Result: **149.3 MB -> 118.7 MB** (1.3x smaller). That is far short of the
   ~4x a fully quantized model should give (compare the *reference* 30 s int8
   export in the same upstream HF repo: 151 MB f32 -> 41 MB i8, a real 3.7x).
   The likely explanation, matching what README.md already says about this
   model: `ai_edge_quantizer`'s default `static_wi8_ai8` recipe only has
   efficient int8 kernels for ops like `FULLY_CONNECTED`/`CONV_2D`; Whisper's
   attention math goes through `BATCH_MATMUL` and a large `EMBEDDING_LOOKUP`
   table (~80 MB alone in the float model), and those either keep float
   weights or aren't covered by the default op list. This could not be
   confirmed by static analysis of the exported flatbuffer -- newer
   LiteRT exports store large constants in an external buffer-offset segment
   that the `tflite` PyPI package's (older) generated schema reads as empty --
   so treat this as the plausible explanation, not a proven one.

5. **Verify the exported model is still loadable on the target's TFLite
   2.15** (see `tflite-215-no-stablehlo`: TFLite 2.15 rejects
   `STABLEHLO_COMPOSITE` ops, which is why `leuconoe`'s export was chosen over
   `litert-community`'s in the first place). Checked the *custom* op codes in
   the flatbuffer (STABLEHLO ops always show up as `CUSTOM`, never as a
   builtin): **none found**, same builtin op codes as the float model. Safe
   to load on-device.

6. **Sanity-check the transcript.** The quantized model's signatures now
   expose **int8 tensors at the model boundary too** (mel input, encoder
   states, decoder logits) -- `static_wi8_ai8` does not keep a float32 I/O
   fallback. Wrote `whisper_bench_src/backend_tflite_int8.py`, a sibling of
   the existing `TFLiteBackend`, that affine-quantizes the mel input and
   dequantizes/re-quantizes the encoder-states hand-off (the encoder's
   `output_0` and the decoder's `args_0` are the same logical tensor but were
   calibrated with **different** `(scale, zero_point)` -- confirmed by
   printing `get_input/output_details()` -- so passing the raw int8 bytes
   through unchanged would silently be wrong). Argmax over the still-int8
   logits is equivalent to argmax over the dequantized floats for affine,
   per-tensor, monotonic quantization, so the decode loop needs no float
   conversion at all for greedy decoding.

   **Result: garbage.** On the same `jfk.wav` that all three existing
   backends (Hailo-8, float NPU, float CPU) transcribe correctly, the int8
   model produces nonsense tokens (mixed CJK/Latin garbage, immediate
   4-gram-repeat stop). This is a known failure mode for naive per-tensor
   static int8 post-training quantization of transformers: attention
   softmax/layer-norm activations have long-tailed outlier distributions
   that a single (scale, zero_point) per tensor cannot represent, and Whisper
   was not quantization-aware trained. It is *not* a bug in the pipeline --
   the model loads, runs, and produces a fixed valid-shaped int8 tensor, it's
   just numerically wrong.

7. **Follow-up tried:** `recipe.static_wi8_ai16()` (int8 weights, **int16**
   activations) -- the standard mitigation for exactly this failure mode,
   since 16-bit activations keep enough dynamic range for attention/LayerNorm
   outliers while still quantizing the weights. Same calibration data, same
   pipeline, same op-set check (clean, no STABLEHLO). Model shrank the same
   149.3 MB -> 119.1 MB (int8 weights dominate the size either way).
   **Also produced a broken transcript** (`!`, immediate degenerate output)
   on the same `jfk.wav`. So widening activations to int16 did not rescue
   this graph either -- the int8 *weights* (or the small, single-recording
   calibration set) are the more likely culprit, not activation range alone.

## Files

All in this `quantization/` directory, next to but not referenced by the
recipe (nothing here is installed to the target):

- `quantize_int8.py` -- the int8/int8 run (steps 1-5 above). Imports
  `../files/whisper_bench` directly (pure numpy/stdlib, safe to run on a
  dev host).
- `quantize_int8_ai16.py` -- identical, with `static_wi8_ai16()` instead.
- `backend_tflite_int8.py` -- `QuantizedTFLiteBackend`, a `Backend`
  implementation for a fully int8-quantized signature model (handles the
  quantize/dequantize boundary transparently; `encode()`/`decode_logits()`
  match `TFLiteBackend`'s interface so it *would* plug into `bench.py` and
  `transcribe.py` unmodified) -- kept as a validated reference, not wired
  into the package; see "Bottom line" below for why.
- the quantized `.tflite` artifacts themselves were produced on a dev host
  under `models/` and are **not** checked in here (2 x ~119 MB, and neither
  one works) -- rerun the scripts to reproduce them.

## Bottom line

int8 (and int8w/int16a) quantization of this model is **technically
achievable and loads cleanly on TFLite 2.15** -- same op set as the float
model, no STABLEHLO, correct shapes -- but **static post-training
quantization destroys Whisper-tiny's output** on this graph in both variants
tried. This is consistent with why the existing recipe deliberately ships
the float32 graph: Whisper-tiny was never quantization-aware trained, and a
single-recording, four-window calibration set is also too narrow for
reliable PTQ ranges. **Do not wire either quantized model into
`demo-whisper-benchmark-data`** -- neither reproduces a correct transcript,
so there is nothing to benchmark against the NPU/CPU/Hailo-8 numbers yet.

## What would be needed to actually get a usable int8 model

Not attempted here (out of scope for this pass -- this was a reproducible
*process* exercise, not an accuracy-tuning one):

- **A much larger, more diverse calibration set** (many speakers, languages,
  SNRs) -- the single 11 s `jfk.wav` sliced four ways is a demonstration
  calibration set, not a representative one.
- **Selective/mixed precision**: quantize `CONV_2D`/`FULLY_CONNECTED` to int8
  (where TFLite has solid int8 kernels) while forcing `BATCH_MATMUL`,
  `SOFTMAX`, and the layer-norm `MEAN`/`RSQRT` chain to stay float32 or int16
  -- `ai_edge_quantizer`'s recipe API supports per-op overrides, not just the
  three global presets used here.
- **Quantization-aware fine-tuning** of Whisper-tiny itself, which is the
  standard fix for transformer PTQ collapse and is a materially bigger
  undertaking (needs the PyTorch/TF source model and a training loop, not
  just a `.tflite` file and a quantizer).
- Only once a variant produces a correct transcript does the original
  motivating question -- does int8 let the VX delegate keep more of the
  graph on the NPU instead of bouncing to the CPU -- become measurable at
  all; it was not reached in this pass.
