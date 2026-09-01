# Whisper Benchmark — Hailo-8 vs. i.MX8MP CPU

Runs OpenAI **Whisper-tiny** speech recognition over the *same* 10 second
audio window on both backends and reports how many inferences per second
each sustains:

| Backend       | Runtime          | Executes on                                   |
|---------------|------------------|-----------------------------------------------|
| `hailo`       | HailoRT 4.23     | Hailo-8 AI accelerator (26 TOPS, PCIe)        |
| `cpu`         | TFLite 2.15      | 4x Cortex-A53 via XNNPACK — the reference bar |

There is deliberately no `npu` backend. Read "Why there is no NPU backend"
below before wondering where it went.

## Running it

```sh
whisper-benchmark                      # both backends
whisper-benchmark -b cpu               # just the CPU reference
whisper-benchmark -a /path/to/16k.wav  # your own 16 kHz WAV
whisper-benchmark --json results.json  # machine readable output
```

Measured on a phyBOARD-Pollux i.MX8MP with a Hailo-8 on PCIe:

```
Whisper-tiny — Hailo-8 vs. i.MX8MP CPU
11.0 s audio, 10 s window fed to both encoders, 0.4 s leading silence trimmed

Backend           enc inf/s   enc ms  dec steps/s   dec ms   full ms  x realtime
--------------------------------------------------------------------------------
Hailo-8               29.39       34        15.71       64      1790       5.59x
i.MX8MP CPU            0.97     1030         1.68      594     17787       0.56x

Speed-up vs. i.MX8MP CPU: Hailo-8 9.9x

Transcripts
  Hailo-8          And so my fellow Americans ask not what your country can do
                   for you, ask what you can do for your country
  i.MX8MP CPU      And so my fellow Americans ask not what your country can do
                   for you, ask what you can do for your country
```

Both print the same sentence — that is the point of showing the transcripts.
If they agree, the throughput figures are being compared on identical work
rather than on one backend quietly doing less.

## Using one backend as plain speech-to-text

`whisper-benchmark` is a comparison harness: it always runs both backends and
reports timing. For actual transcription, two standalone commands wrap the
same code around a single backend and just print the transcript:

```sh
whisper-hailo  recording.wav   # Hailo-8
whisper-cpu    recording.wav   # i.MX8MP CPU (XNNPACK)
```

Audio longer than the compiled window (10 s) is split into consecutive
chunks and each chunk's transcript is concatenated; there is no cross-chunk
context, so a sentence split across a chunk boundary can come out
imperfectly, and a near-silent trailing chunk can make greedy Whisper
hallucinate a filler phrase (e.g. `[INAUDIBLE]`) — both are properties of
Whisper-tiny's fixed-window decoder, not of this wrapper. Run with no
argument to transcribe the bundled sample; see `whisper-hailo --help` for
the full option list (data directory, thread count, `-q` to suppress
progress and print only the transcript).

**The Hailo-8 is ~30x the CPU on the encoder and ~10x end-to-end**, and it is
the difference between 5.6x realtime (comfortably faster than a live
microphone) and 0.56x (compute takes longer than the speech itself).

## Speaking into it live

`whisper-benchmark`, `whisper-hailo` and `whisper-cpu` all accept `-m`/`--mic`
in place of a WAV file: it records through `sounddevice`/PortAudio, press
Enter to stop early, otherwise stop after the given maximum (default: the
10 s compiled window), then transcribes that instead. This matches the
interaction model of Hailo's own `hailo-apps` `speech_recognition` reference
tool (press Enter to start/stop) rather than a fixed-duration capture.

```sh
whisper-hailo --mic                       # record up to 10 s from the default device, Enter to stop early
whisper-hailo --mic 5                     # cap at 5 s instead
python3 -m sounddevice                    # list input devices if the default one is wrong
whisper-hailo --mic -D USB                # select by name substring
whisper-hailo --mic -D 1                  # or by device index
whisper-benchmark --mic                   # live mic, both backends compared
```

Validated on a phyBOARD-Pollux i.MX8MP with a Hailo-8 and a USB microphone
(Jabra Speak 510): `whisper-hailo --mic -D USB`. Run non-interactively (no
TTY on stdin, e.g. a plain `ssh host whisper-hailo --mic`) it cannot detect
Enter and simply records for the full maximum duration.

There is no on-chip live-microphone mode here comparable to the Hailo-10H
GenAI `Speech2Text` API (`hailo_platform.genai`) — that high-level API only
exists on Hailo-10H. Hailo-8 speech-to-text goes through the same low-level
HailoRT `InferModel` API with separate encoder/decoder HEFs that the GenAI
API itself uses internally (and that Hailo's own cross-device
`hailo-apps`/`hailo-whisper` tools use for Hailo-8), so `-m/--mic` gets to
the same place (microphone in, transcript out) by recording a WAV first and
feeding it through the existing pipeline. Requires `python3-sounddevice`
(pulls in `portaudio-v19` for `libportaudio`) on the target.

## What is measured

* **enc inf/s** — complete Whisper-tiny encoder passes over a 10 s Mel
  spectrogram, per second. This is the headline number: both backends run
  the identical workload, so the figures are directly comparable.
* **dec steps/s** — decoder passes per second. One generated token costs one
  full decoder pass on both devices; neither compiled graph carries a KV cache.
* **full ms** — encode plus the complete greedy decode of the window, i.e. what
  an application actually waits for.
* **x realtime** — seconds of audio transcribed per second of wall clock.
  Anything above 1.0x can keep up with a live microphone.

All figures are medians over repeated runs after an explicit warm-up,
reported separately as *first inference* rather than being averaged in.

## Models

| File                                                   | Source                            |
|--------------------------------------------------------|-----------------------------------|
| `tiny-whisper-encoder-10s_15dB.hef`                     | Hailo model resources, Hailo-8    |
| `tiny-whisper-decoder-fixed-sequence-matmul-split.hef`  | Hailo model resources, Hailo-8    |
| `whisper_tiny_10s_f32.tflite`                           | LiteRT export of `openai/whisper-tiny` |
| `mel_filters.npz`, `vocab.json`                         | OpenAI Whisper front-end assets   |
| `jfk.wav`                                               | whisper.cpp sample, public domain |

Everything lives under `/usr/share/demo-whisper-benchmark`; override the
location with `--data-dir`.

Two details of the compiled models differ and are worth knowing when reading
the decoder column:

* The Hailo-8 decoder HEF holds **32** token positions, the TFLite decoder
  **128**. Cost per decoder pass therefore is not identical work — the encoder
  column is the clean comparison.
* The Hailo-8 decoder HEF was compiled without the token embedding lookup, so
  the gather and the positional add run on the CPU. The TFLite graph keeps
  `EMBEDDING_LOOKUP` inside the model. Both are properties of the published
  models, not of this demo.

## Why there is no NPU backend

There used to be one. Measured, the i.MX8MP NPU (VeriSilicon VIP8000,
2.3 TOPS) was **5.3x slower than the plain Cortex-A53 CPU** on this exact
model — not a misconfiguration: NXP's VX delegate reported
`error_during_init / error_during_prepare / error_during_invoke = 0` and
wrote ~230 MB of compiled graph binaries into a cache, so it genuinely
executed the model, just slowly. It was removed rather than kept as a
"technically works" option, because shipping a working-but-much-worse `-b
npu` choice would suggest the NPU is a viable path for a workload like this,
which it is not, and that is the wrong takeaway for anyone using this demo
to evaluate the AI-Kit.

**Why**: the VIP8000 is built for quantised CNNs. Whisper is a transformer,
and its `BATCH_MATMUL`, `TRANSPOSE` and `MEAN`+`RSQRT` layer-norm chains
either map poorly onto the NPU's fp16 path or get handed back to the CPU
with a tensor layout conversion in each direction. 2.3 TOPS on paper does
not help when the graph does not fit the hardware's shape. This matches
NXP's own eIQ Model Zoo: its speech-recognition example for this NPU is
[wav2letter](https://github.com/NXP/eiq-model-zoo/tree/main/tasks/audio/speech-recognition/wav2letter),
a CNN acoustic model, not a transformer — NXP did not ship a Whisper example
for this NPU either.

We also tried to fix it with post-training int8 quantization (weights and
activations, and separately weights-int8/activations-int16) using
`ai-edge-quantizer` — see `../quantization/QUANTIZATION.md` for the full,
reproducible protocol. Both quantized variants loaded fine but produced
garbage transcripts: naive static PTQ collapses transformer attention
without quantization-aware training, so quantization is not a shortcut
around the architecture mismatch either.

This is exactly the argument for the AI-Kit: for CNN inference the on-SoC
NPU is the right tool, but for a transformer workload like ASR you want a
dedicated accelerator that supports it. The `cpu` row stays in the table so
that claim is falsifiable rather than asserted — Hailo-8 vs. Hailo-8 alone
would just be a number with nothing to compare it to.
