"""Command line front-end for the Whisper Hailo-8 vs i.MX8MP CPU benchmark."""

import argparse
import os
import sys
import time

from . import audio as audio_module
from . import mic as mic_module
from . import report as report_module
from .bench import Result, run_backend
from .tokenizer import WhisperTokenizer, clean_transcription

DEFAULT_DATA_DIR = "/usr/share/demo-whisper-benchmark"

BACKEND_CHOICES = ("hailo", "cpu")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="whisper-benchmark",
        description="Compare Whisper-tiny inference throughput on the "
                    "Hailo-8 AI accelerator and the i.MX8MP Cortex-A53 cores.",
    )
    parser.add_argument(
        "-b", "--backend", action="append", choices=BACKEND_CHOICES,
        metavar="{hailo,cpu}",
        help="backend to measure; repeatable (default: both)",
    )
    parser.add_argument(
        "-a", "--audio", metavar="WAV",
        help="16 kHz WAV file to transcribe (default: the bundled sample)",
    )
    parser.add_argument(
        "-m", "--mic", nargs="?", type=float, const=float(audio_module.CHUNK_LENGTH),
        metavar="SECONDS",
        help="record from the microphone instead of a file -- press Enter to "
             f"stop early, otherwise stop after SECONDS (default: "
             f"{audio_module.CHUNK_LENGTH}); mutually exclusive with --audio",
    )
    parser.add_argument(
        "-D", "--device", default=mic_module.DEFAULT_DEVICE, metavar="DEVICE",
        help="input device for --mic: an index or a name substring from "
             "'python3 -m sounddevice' (default: system default)",
    )
    parser.add_argument(
        "-d", "--data-dir", default=DEFAULT_DATA_DIR, metavar="DIR",
        help=f"model and asset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "-n", "--repeats", type=int, default=10, metavar="N",
        help="encoder inferences to time per backend (default: 10)",
    )
    parser.add_argument(
        "--decode-repeats", type=int, default=20, metavar="N",
        help="decoder steps to time per backend (default: 20)",
    )
    parser.add_argument(
        "--e2e-repeats", type=int, default=2, metavar="N",
        help="full transcriptions to time per backend (default: 2)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=os.cpu_count() or 4, metavar="N",
        help="CPU threads for the TFLite backend (default: all cores)",
    )
    parser.add_argument(
        "--no-trim", action="store_true",
        help="do not trim leading silence before the 10 s window",
    )
    parser.add_argument(
        "--json", metavar="FILE", nargs="?", const="-",
        help="write the raw measurements as JSON ('-' for stdout)",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="disable ANSI colour",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="only print the report",
    )
    return parser


def _asset(data_dir, *parts):
    path = os.path.join(data_dir, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _make_backend(kind, args):
    """Construct one backend. Imports are local so a missing runtime only
    disables that backend instead of the whole demo."""
    if kind == "hailo":
        from .backend_hailo import HailoBackend
        return HailoBackend(
            encoder_path=_asset(args.data_dir, "hailo",
                                "tiny-whisper-encoder-10s_15dB.hef"),
            decoder_path=_asset(args.data_dir, "hailo",
                                "tiny-whisper-decoder-fixed-sequence-matmul-split.hef"),
            embedding_path=_asset(args.data_dir, "hailo",
                                  "token_embedding_weight_tiny.npy"),
            positional_path=_asset(args.data_dir, "hailo",
                                   "onnx_add_input_tiny.npy"),
            chunk_length=audio_module.CHUNK_LENGTH,
        )

    from .backend_tflite import TFLiteBackend
    return TFLiteBackend(
        model_path=_asset(args.data_dir, "tflite", "whisper_tiny_10s_f32.tflite"),
        num_threads=args.threads,
        chunk_length=audio_module.CHUNK_LENGTH,
    )


LABELS = {
    "hailo": "Hailo-8",
    "cpu": "i.MX8MP CPU",
}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mic is not None and args.audio:
        parser.error("argument -a/--audio: not allowed with --mic")
    backends = args.backend or list(BACKEND_CHOICES)
    style = report_module.Style(
        enabled=not args.no_color and sys.stdout.isatty()
    )

    def log(message):
        if not args.quiet:
            print(message, file=sys.stderr)

    try:
        filters = audio_module.load_mel_filters(
            _asset(args.data_dir, "assets", "mel_filters.npz"))
        tokenizer = WhisperTokenizer(
            _asset(args.data_dir, "assets", "vocab.json"))
        default_wav = None if args.mic is not None else (
            args.audio or _asset(args.data_dir, "assets", "jfk.wav"))
    except FileNotFoundError as error:
        print(f"missing demo data: {error}\n"
              f"install demo-whisper-benchmark-data or pass --data-dir",
              file=sys.stderr)
        return 2

    recorded_path = None
    if args.mic is not None:
        try:
            recorded_path = mic_module.record_wav(
                args.mic, device=args.device, prompt=log)
        except RuntimeError as error:
            print(f"microphone capture failed: {error}", file=sys.stderr)
            return 1
    wav_path = recorded_path or default_wav

    try:
        mel, audio_info = audio_module.prepare_window(
            wav_path, filters, trim_silence=not args.no_trim)
        log(f"audio: {os.path.basename(wav_path)} "
            f"({audio_info['duration']:.1f} s), mel {mel.shape}")
    finally:
        if recorded_path:
            os.remove(recorded_path)

    results = []
    for kind in backends:
        label = LABELS[kind]
        log(f"\n[{label}] loading")
        try:
            backend, load_seconds = _timed(_make_backend, kind, args)
        except Exception as error:  # noqa: BLE001 - report and keep going
            log(f"[{label}] unavailable: {error}")
            results.append(Result(
                name=label, device=kind, details={}, chunk_length=0.0,
                error=str(error),
            ))
            continue

        backend.name = label
        try:
            with backend:
                result = run_backend(
                    backend, mel, tokenizer,
                    repeats=args.repeats,
                    decode_repeats=args.decode_repeats,
                    e2e_repeats=args.e2e_repeats,
                    progress=lambda msg: log(f"[{label}] {msg}"),
                )
            result.load_seconds = load_seconds
            result.transcript = clean_transcription(result.transcript)
            results.append(result)
        except Exception as error:  # noqa: BLE001
            log(f"[{label}] failed: {error}")
            results.append(Result(
                name=label, device=kind, details={}, chunk_length=0.0,
                error=str(error),
            ))

    print(report_module.render(results, audio_info, style))

    if args.json:
        payload = report_module.to_json(results, audio_info)
        if args.json == "-":
            print(payload)
        else:
            with open(args.json, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
            log(f"wrote {args.json}")

    return 0 if any(not r.error for r in results) else 1


def _timed(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - start


if __name__ == "__main__":
    sys.exit(main())
