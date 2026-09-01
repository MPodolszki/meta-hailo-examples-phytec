# python3-sounddevice (whisper-hailo/whisper-cpu/whisper-benchmark --mic)
# only needs the ALSA backend. The upstream default also builds against
# JACK, which then makes libportaudio.so.2 dlopen() libjack.so.0 at runtime
# -- pulling in a full JACK audio server for a demo that never uses it.
PACKAGECONFIG:remove = "jack"
