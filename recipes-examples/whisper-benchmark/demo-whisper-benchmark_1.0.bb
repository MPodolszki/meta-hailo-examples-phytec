SUMMARY = "Whisper speech recognition: Hailo-8 vs i.MX8MP CPU, benchmark and standalone STT"
DESCRIPTION = "Runs the same Whisper-tiny model over the same 10 s audio \
               window on the Hailo-8 AI accelerator (HailoRT) and on the \
               i.MX8MP Cortex-A53 cores (TensorFlow Lite/XNNPACK) as a \
               reference, and reports encoder inferences per second, decoder \
               steps per second and the realtime factor side by side. Also \
               installs whisper-hailo and whisper-cpu: independent \
               speech-to-text commands that run Whisper-tiny on one backend \
               and print a transcript of a WAV file of any length, chunked \
               into the backend's fixed window, or of a live microphone \
               recording via -m/--mic (press Enter to stop early, via \
               sounddevice/PortAudio). \
               \
               There is deliberately no i.MX8MP NPU backend: measured, the \
               NPU (VeriSilicon VIP8000) is slower than the CPU on this \
               model, because it is a quantized-CNN engine and Whisper's \
               attention/layernorm ops don't map onto it -- see README.md."
HOMEPAGE = "https://www.phytec.de"

LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://whisper_bench \
    file://whisper-benchmark \
    file://whisper-hailo \
    file://whisper-cpu \
    file://README.md \
"

S = "${WORKDIR}"

inherit python3-dir

do_configure[noexec] = "1"
do_compile[noexec] = "1"

do_install() {
    install -d ${D}${PYTHON_SITEPACKAGES_DIR}/whisper_bench
    install -m 0644 ${WORKDIR}/whisper_bench/*.py ${D}${PYTHON_SITEPACKAGES_DIR}/whisper_bench/

    install -d ${D}${bindir}
    install -m 0755 ${WORKDIR}/whisper-benchmark ${D}${bindir}/whisper-benchmark
    install -m 0755 ${WORKDIR}/whisper-hailo ${D}${bindir}/whisper-hailo
    install -m 0755 ${WORKDIR}/whisper-cpu ${D}${bindir}/whisper-cpu

    install -d ${D}${docdir}/${BPN}
    install -m 0644 ${WORKDIR}/README.md ${D}${docdir}/${BPN}/
}

FILES:${PN} = " \
    ${bindir}/whisper-benchmark \
    ${bindir}/whisper-hailo \
    ${bindir}/whisper-cpu \
    ${PYTHON_SITEPACKAGES_DIR}/whisper_bench \
    ${docdir}/${BPN} \
"

# backend_hailo.py's HEFs are compiled specifically for Hailo-8 (see whisper_bench/backend_hailo.py),
# so this demo always needs the Hailo-8 Python bindings - it's only pulled into
# phytec-hailo-image.bb when HAILO_CHIP = "hailo8" (the default).
RDEPENDS:${PN} += " \
    demo-whisper-benchmark-data \
    hailo8-python-wheels \
    python3-audio \
    python3-core \
    python3-json \
    python3-numpy \
    tensorflow-lite \
"

# The demo degrades gracefully: a backend whose runtime is missing is reported
# as unavailable and the remaining ones are still measured. python3-sounddevice
# (and the libportaudio it dlopens, via its portaudio-v19 RDEPENDS) is only
# needed for -m/--mic, hence recommend rather than require it.
RRECOMMENDS:${PN} += "hailo8-firmware libhailort hailo8-pci python3-sounddevice"

COMPATIBLE_MACHINE = "(mx8mp-nxp-bsp)"
