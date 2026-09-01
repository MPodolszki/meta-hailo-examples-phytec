"""Standalone speech-to-text front-end: one backend, one transcript.

Unlike whisper-benchmark, this does not time or compare backends -- it is the
plain way to run Whisper-tiny speech-to-text on this board with a single
accelerator. whisper-hailo and whisper-cpu each call run() with a fixed
backend so every accelerator is an independent STT command.
"""

import argparse
import os
import sys

from . import audio as audio_module
from . import mic as mic_module
from .cli import DEFAULT_DATA_DIR, _asset, _make_backend
from .tokenizer import WhisperTokenizer
from .transcribe import transcribe


def build_parser(prog):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Whisper-tiny speech-to-text.",
    )
    parser.add_argument(
        "audio", nargs="?", metavar="WAV",
        help="16 kHz WAV file to transcribe (default: the bundled sample)",
    )
    parser.add_argument(
        "-m", "--mic", nargs="?", type=float, const=float(audio_module.CHUNK_LENGTH),
        metavar="SECONDS",
        help="record from the microphone instead of a file -- press Enter to "
             f"stop early, otherwise stop after SECONDS (default: "
             f"{audio_module.CHUNK_LENGTH}); mutually exclusive with WAV",
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
        "-t", "--threads", type=int, default=os.cpu_count() or 4, metavar="N",
        help="CPU threads for the TFLite backend (default: all cores)",
    )
    parser.add_argument(
        "--no-trim", action="store_true",
        help="do not trim leading silence",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="print only the transcript, no progress on stderr",
    )
    return parser


def run(backend_kind, argv=None):
    """Entry point shared by whisper-hailo / whisper-cpu."""
    parser = build_parser(prog=f"whisper-{backend_kind}")
    args = parser.parse_args(argv)

    if args.mic is not None and args.audio:
        parser.error("argument WAV: not allowed with --mic")

    def log(message):
        if not args.quiet:
            print(message, file=sys.stderr)

    try:
        filters = audio_module.load_mel_filters(
            _asset(args.data_dir, "assets", "mel_filters.npz"))
        tokenizer = WhisperTokenizer(_asset(args.data_dir, "assets", "vocab.json"))
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

    log(f"loading {backend_kind}")
    try:
        backend = _make_backend(backend_kind, args)
    except Exception as error:  # noqa: BLE001 - report a clean CLI error
        print(f"{backend_kind} unavailable: {error}", file=sys.stderr)
        if recorded_path:
            os.remove(recorded_path)
        return 1

    backend.name = backend_kind
    try:
        with backend:
            text, info = transcribe(
                backend, wav_path, filters, tokenizer,
                trim_silence=not args.no_trim,
                progress=lambda msg: log(f"[{backend_kind}] {msg}"),
            )
    except Exception as error:  # noqa: BLE001
        print(f"transcription failed: {error}", file=sys.stderr)
        return 1
    finally:
        if recorded_path:
            os.remove(recorded_path)

    log(f"{info['duration']:.1f}s audio, {info['chunks']} chunk(s) of "
        f"{info['chunk_length']:.0f}s")
    print(text)
    return 0
