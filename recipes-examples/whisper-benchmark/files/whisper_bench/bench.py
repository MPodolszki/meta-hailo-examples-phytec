"""Timing harness.

Everything is measured as steady-state latency after an explicit warm-up,
reported separately from the throughput figure rather than averaged in.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .tokenizer import FORCED_DECODER_IDS


@dataclass
class Timing:
    """Latency statistics for one repeatedly executed operation."""

    samples: List[float] = field(default_factory=list)

    @property
    def count(self):
        return len(self.samples)

    @property
    def mean(self):
        return float(np.mean(self.samples)) if self.samples else float("nan")

    @property
    def median(self):
        return float(np.median(self.samples)) if self.samples else float("nan")

    @property
    def best(self):
        return float(np.min(self.samples)) if self.samples else float("nan")

    @property
    def worst(self):
        return float(np.max(self.samples)) if self.samples else float("nan")

    @property
    def per_second(self):
        """Operations per second, derived from the *median* latency.

        Median rather than mean: on a loaded embedded target a single
        scheduling hiccup or a DVFS ramp in the first iteration skews the mean
        badly, and the number quoted for a demo should be the one the board
        actually sustains.
        """
        median = self.median
        return 1.0 / median if median == median and median > 0 else float("nan")

    def as_dict(self):
        return {
            "count": self.count,
            "mean_ms": self.mean * 1000.0,
            "median_ms": self.median * 1000.0,
            "best_ms": self.best * 1000.0,
            "worst_ms": self.worst * 1000.0,
            "per_second": self.per_second,
        }


@dataclass
class Result:
    """Everything measured for one backend."""

    name: str
    device: str
    details: Dict[str, object]
    chunk_length: float
    load_seconds: float = 0.0
    warmup_seconds: float = 0.0
    encoder: Timing = field(default_factory=Timing)
    decoder: Timing = field(default_factory=Timing)
    transcription: Timing = field(default_factory=Timing)
    transcript: str = ""
    tokens: int = 0
    stop_reason: str = ""
    error: Optional[str] = None

    @property
    def realtime_factor(self):
        """Seconds of audio transcribed per second of wall clock."""
        median = self.transcription.median
        return self.chunk_length / median if median == median and median > 0 \
            else float("nan")

    def as_dict(self):
        return {
            "backend": self.name,
            "device": self.device,
            "details": self.details,
            "chunk_length_s": self.chunk_length,
            "load_seconds": self.load_seconds,
            "warmup_seconds": self.warmup_seconds,
            "encoder": self.encoder.as_dict(),
            "decoder": self.decoder.as_dict(),
            "transcription": self.transcription.as_dict(),
            "realtime_factor": self.realtime_factor,
            "transcript": self.transcript,
            "tokens": self.tokens,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


def _time(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - start


def run_backend(backend, mel, tokenizer, repeats=10, decode_repeats=20,
                e2e_repeats=2, progress=None):
    """Benchmark one backend on a single prepared Mel window."""
    result = Result(
        name=backend.name,
        device=backend.device,
        details=backend.describe(),
        chunk_length=float(backend.chunk_length),
    )

    def note(message):
        if progress:
            progress(message)

    note("warm-up")
    encoded, result.warmup_seconds = _time(backend.encode, mel)

    # Transcribe once to get the reference transcript and the token count the
    # decoder phase replays. Run it as a full encode + decode so the pass is
    # itself a valid end-to-end sample and can be kept rather than thrown away.
    note("transcribing")
    (tokens, stop_reason), elapsed = _time(_transcribe_once, backend, mel)
    result.transcription.samples.append(elapsed)
    result.tokens = len(tokens)
    result.stop_reason = stop_reason
    result.transcript = tokenizer.decode(tokens)

    # Encoder throughput.
    note(f"encoder x{repeats}")
    for _ in range(repeats):
        encoded, elapsed = _time(backend.encode, mel)
        result.encoder.samples.append(elapsed)

    # Decoder throughput: one decoder pass per generated token. Replay the
    # tokens that were actually produced so the measured work is the work the
    # transcription really did.
    note(f"decoder x{decode_repeats}")
    token_ids = np.zeros((1, backend.sequence_length), dtype=np.int64)
    prompt = list(FORCED_DECODER_IDS) + tokens
    for index, token in enumerate(prompt[:backend.sequence_length]):
        token_ids[0][index] = token

    first = len(FORCED_DECODER_IDS) - 1
    last = min(first + max(result.tokens, 1), backend.sequence_length - 1)
    positions = list(range(first, last)) or [first]

    for step in range(decode_repeats):
        position = positions[step % len(positions)]
        _, elapsed = _time(backend.decode_logits, encoded, token_ids, position)
        result.decoder.samples.append(elapsed)

    # End-to-end: encode + full greedy decode, exactly as an application runs
    # it. The transcribe pass above already contributed one sample.
    extra = max(e2e_repeats - len(result.transcription.samples), 0)
    if extra:
        note(f"end-to-end x{extra}")
        for _ in range(extra):
            _, elapsed = _time(_transcribe_once, backend, mel)
            result.transcription.samples.append(elapsed)

    return result


def _transcribe_once(backend, mel):
    encoded = backend.encode(mel)
    return backend.generate(encoded)
