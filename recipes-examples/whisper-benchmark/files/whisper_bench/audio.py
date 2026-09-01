"""Whisper audio front-end: WAV loading and log-Mel spectrogram.

Deliberately NumPy-only. The upstream Hailo/OpenAI implementations use
torch.stft, but pulling PyTorch onto the target just for a 400-point FFT is
not worth ~1 GB of rootfs. The output of log_mel_spectrogram() matches the
torch implementation to within float32 rounding.
"""

import wave

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

# Both backends run a 10 s window: the Hailo HEF is compiled for 10 s and the
# TFLite encoder signature takes [1, 80, 1000] == 10 s at hop 160.
CHUNK_LENGTH = 10
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH


def load_wav(path):
    """Read a WAV file as mono float32 in [-1, 1], resampled to 16 kHz."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != SAMPLE_RATE:
        audio = _resample_linear(audio, rate, SAMPLE_RATE)

    return np.ascontiguousarray(audio, dtype=np.float32)


def _resample_linear(audio, src_rate, dst_rate):
    """Linear resampling. Good enough for a benchmark's reference audio."""
    duration = len(audio) / src_rate
    dst_len = int(round(duration * dst_rate))
    src_idx = np.linspace(0, len(audio) - 1, dst_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(len(audio)), audio).astype(np.float32)


def normalize_peak(audio, target_peak=0.9):
    """Peak-normalize the waveform.

    Whisper was trained on properly levelled audio; a quiet recording makes the
    encoder treat the signal as silence and the decoder emits garbage. Both
    backends get the identical normalized waveform so the comparison is fair.
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-6:
        audio = audio * (target_peak / peak)
    return audio


def detect_speech_start(audio, threshold=0.2, frame_duration=0.2):
    """Return the time in seconds of the first frame that carries speech.

    A 10 s window is short enough that leading silence costs real transcript.
    Energy-based onset detection is what Hailo's reference pipeline uses too.
    """
    frame_size = max(int(frame_duration * SAMPLE_RATE), 1)
    frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]
    energies = np.array([float(np.mean(frame ** 2)) for frame in frames if frame.size])
    if energies.size == 0 or energies.max() <= 0:
        return 0.0

    energies /= energies.max()
    speech = np.where(energies > threshold)[0]
    if not speech.size:
        return 0.0

    # Back off one frame: the onset sits somewhere inside the first loud frame,
    # and clipping the leading consonant costs a word in the transcript.
    return max(float((speech[0] - 1) * frame_duration), 0.0)


def pad_or_trim(audio, length=N_SAMPLES):
    if len(audio) > length:
        return audio[:length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    return audio


def load_mel_filters(path, n_mels=N_MELS):
    """Load the Mel filterbank shipped with the demo data (OpenAI asset)."""
    with np.load(str(path), allow_pickle=False) as data:
        return np.asarray(data[f"mel_{n_mels}"], dtype=np.float32)


def log_mel_spectrogram(audio, filters):
    """Whisper log-Mel spectrogram, shape [n_mels, frames].

    Mirrors openai/whisper: centred reflect-padded STFT with a periodic Hann
    window, power magnitudes, Mel projection, log10, then the fixed dynamic
    range clamp and (x + 4) / 4 scaling.
    """
    audio = np.asarray(audio, dtype=np.float32)

    # torch.hann_window() is periodic, unlike np.hanning().
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)

    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    strides = (padded.strides[0] * HOP_LENGTH, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=(n_frames, N_FFT), strides=strides
    )

    spectrum = np.fft.rfft(frames * window, n=N_FFT, axis=-1)
    # Drop the final frame, as torch.stft(...)[..., :-1] does.
    magnitudes = np.abs(spectrum[:-1]) ** 2

    mel_spec = filters @ magnitudes.T.astype(np.float32)
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def prepare_window(path, filters, chunk_length=CHUNK_LENGTH, trim_silence=True):
    """Load *path* and turn it into the single Mel window both backends see.

    Returns (mel, info) where mel is [n_mels, chunk_length * 100].
    """
    audio = load_wav(path)
    duration = len(audio) / SAMPLE_RATE

    offset = detect_speech_start(audio) if trim_silence else 0.0
    if offset > 0:
        audio = audio[int(offset * SAMPLE_RATE):]

    audio = normalize_peak(audio)
    window = pad_or_trim(audio, chunk_length * SAMPLE_RATE)

    info = {
        "duration": duration,
        "speech_offset": offset,
        "window": float(chunk_length),
    }
    return log_mel_spectrogram(window, filters), info
