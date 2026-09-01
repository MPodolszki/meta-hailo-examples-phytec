"""Standalone speech-to-text: chunked greedy transcription for one backend.

bench.py times a single fixed backend.chunk_length window for the benchmark.
This module is the "real" front-end: it splits audio of any length into
consecutive chunk_length windows and concatenates the transcript of each, so
any of the three backends can be used as an actual speech-to-text tool
instead of only inside the benchmark comparison.
"""

from . import audio as audio_module
from .tokenizer import clean_transcription


def transcribe(backend, wav_path, filters, tokenizer, trim_silence=True,
               progress=None):
    """Transcribe an entire WAV file with *backend*.

    Leading silence is trimmed once, from the start of the whole recording,
    not per chunk -- trimming inside a chunk would just shift speech into the
    pad region of an already fixed-length window instead of removing it.
    """
    chunk_length = backend.chunk_length
    audio = audio_module.load_wav(wav_path)
    duration = len(audio) / audio_module.SAMPLE_RATE

    offset = audio_module.detect_speech_start(audio) if trim_silence else 0.0
    if offset > 0:
        audio = audio[int(offset * audio_module.SAMPLE_RATE):]

    audio = audio_module.normalize_peak(audio)

    chunk_samples = int(chunk_length * audio_module.SAMPLE_RATE)
    total_chunks = max(1, -(-len(audio) // chunk_samples))

    pieces = []
    for index in range(total_chunks):
        start = index * chunk_samples
        window = audio_module.pad_or_trim(
            audio[start:start + chunk_samples], chunk_samples)

        if progress:
            progress(f"chunk {index + 1}/{total_chunks}")

        mel = audio_module.log_mel_spectrogram(window, filters)
        encoded = backend.encode(mel)
        tokens, _stop_reason = backend.generate(encoded)
        text = clean_transcription(tokenizer.decode(tokens))
        if text:
            pieces.append(text)

    info = {
        "duration": duration,
        "speech_offset": offset,
        "chunk_length": float(chunk_length),
        "chunks": total_chunks,
    }
    return " ".join(pieces), info
