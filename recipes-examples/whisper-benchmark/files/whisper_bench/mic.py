"""Microphone capture via sounddevice/PortAudio, with press-Enter-to-stop.

Matches the interaction model of Hailo's own hailo-apps speech_recognition
reference tool (press Enter to stop early, otherwise stop after a maximum
duration) -- the tool that was validated as working, live-microphone
end-to-end, on this exact board.

Hailo-8 has no on-chip GenAI Speech2Text (hailo_platform.genai.Speech2Text
is Hailo-10H only) -- but this project already runs Whisper through the
same low-level HailoRT InferModel API with separate encoder/decoder HEFs
that the GenAI path (and hailo-apps) use, so a live-microphone front end
only needs a WAV file in front of transcribe(); audio.py/transcribe.py
stay untouched.
"""

import os
import select
import sys
import tempfile
import wave

import numpy as np
import sounddevice as sd

from . import audio as audio_module

DEFAULT_DEVICE = None  # sounddevice/PortAudio default input device
POLL_SECONDS = 0.1


def resolve_device(device):
    """Turn a --device string into what sounddevice expects.

    A plain integer selects a device index (see `python3 -m sounddevice`);
    anything else is passed through as a name substring, which is how
    sounddevice itself resolves device names.
    """
    if device is None or device == "":
        return None
    return int(device) if str(device).isdigit() else device


def _enter_pressed():
    """Non-blocking check for a pending Enter keypress on stdin.

    Returns False (never blocks, never stops early) when stdin is not a
    terminal -- e.g. run over a plain `ssh host cmd` -- so the recording
    falls back to running for the full max_duration.
    """
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    sys.stdin.readline()
    return True


def record_wav(max_duration, device=DEFAULT_DEVICE, rate=audio_module.SAMPLE_RATE,
                channels=1, prompt=lambda message: None):
    """Record from *device* until Enter is pressed or *max_duration* seconds pass.

    Returns the path to a temporary 16-bit PCM WAV file; the caller owns it
    and must remove it when done.
    """
    prompt(f"recording -- press Enter to stop (max {max_duration:.0f}s)")
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    try:
        with sd.InputStream(samplerate=rate, channels=channels, dtype="float32",
                             device=resolve_device(device), callback=callback):
            elapsed = 0.0
            while elapsed < max_duration:
                sd.sleep(int(POLL_SECONDS * 1000))
                elapsed += POLL_SECONDS
                if _enter_pressed():
                    prompt("stopped early")
                    break
    except sd.PortAudioError as error:
        raise RuntimeError(
            f"microphone capture failed: {error}; "
            f"run 'python3 -m sounddevice' to list input devices"
        ) from error

    if not chunks:
        raise RuntimeError("no audio captured")

    audio = np.concatenate(chunks, axis=0)
    audio = audio.reshape(-1) if channels == 1 else audio.mean(axis=1)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")

    fd, path = tempfile.mkstemp(prefix="whisper-mic-", suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            wav_file.writeframes(pcm.tobytes())
    except Exception:
        os.remove(path)
        raise
    return path
