SUMMARY = "Models and assets for the Whisper Hailo-8 vs i.MX8MP CPU benchmark"
DESCRIPTION = "Collects everything the whisper-benchmark demo needs at runtime: \
               the Whisper-tiny HEFs compiled for the Hailo-8, the matching \
               TFLite encoder/decoder graph for the i.MX8MP CPU reference, \
               the Whisper Mel filterbank and vocabulary, and a reference \
               speech sample."
HOMEPAGE = "https://www.phytec.de"

# Whisper itself and the whisper.cpp sample are MIT; the converted TFLite graph
# and the Whisper vocabulary are redistributed under Apache-2.0.
LICENSE = "MIT & Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302 \
                    file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

HAILO_RESOURCES = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources"

# Whisper-tiny for the Hailo-8: separate encoder and decoder HEFs, both
# compiled for a 10 s window. The two .npy files are the token embedding table
# and the positional bias that the H8 decoder HEF leaves to the host.
SRC_URI = " \
    ${HAILO_RESOURCES}/whisper/h8/tiny-whisper-encoder-10s_15dB.hef;name=hef-encoder \
    ${HAILO_RESOURCES}/whisper/h8/tiny-whisper-decoder-fixed-sequence-matmul-split.hef;name=hef-decoder \
    ${HAILO_RESOURCES}/npy%20files/whisper/decoder_assets/tiny/decoder_tokenization/token_embedding_weight_tiny.npy;name=embedding;downloadfilename=token_embedding_weight_tiny.npy \
    ${HAILO_RESOURCES}/npy%20files/whisper/decoder_assets/tiny/decoder_tokenization/onnx_add_input_tiny.npy;name=positional;downloadfilename=onnx_add_input_tiny.npy \
"

# The same Whisper-tiny, exported to a single TFLite file with an 'encode' and
# a 'decode' signature, also for a 10 s window so the CPU reference runs
# exactly the same amount of work per inference as the Hailo-8. There used to
# be an i.MX8MP NPU backend sharing this file (kept float32 for the VX
# delegate's fp16 path); it was removed after measuring the NPU at 5.3x
# *slower* than this same CPU path -- see README.md.
SRC_URI += " \
    https://huggingface.co/leuconoe/whisper-tiny-litert/resolve/${WHISPER_TFLITE_REV}/whisper_tiny_10s_f32.tflite;name=tflite;downloadfilename=whisper_tiny_10s_f32.tflite \
"

# Whisper front-end assets and the reference sample (JFK inaugural address,
# public domain recording, 11 s, 16 kHz mono).
SRC_URI += " \
    https://raw.githubusercontent.com/hailo-ai/hailo-apps/${HAILO_APPS_REV}/hailo_apps/python/standalone_apps/speech_recognition/assets/mel_filters.npz;name=mel;downloadfilename=whisper-mel_filters.npz \
    https://huggingface.co/openai/whisper-tiny/resolve/${WHISPER_TINY_REV}/vocab.json;name=vocab;downloadfilename=whisper-tiny-vocab.json \
    https://raw.githubusercontent.com/ggml-org/whisper.cpp/${WHISPER_CPP_REV}/samples/jfk.wav;name=sample;downloadfilename=whisper-jfk.wav \
"

WHISPER_TFLITE_REV = "89cb4079401993912e02371838feca156c9e013e"
HAILO_APPS_REV = "891ce701c2ebe239a5d277759eb75a30f76678a9"
WHISPER_TINY_REV = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
WHISPER_CPP_REV = "1fe009caeda75f69bc864d6370b10674e45a92bd"

SRC_URI[hef-encoder.sha256sum] = "7f16242531c702a7aa4480e434823467e6a0df5ca9b10827cb3917daaf6aded1"
SRC_URI[hef-decoder.sha256sum] = "b7d87dd94c152a35484f7cdb30c011a9e92b6dcadb58d53df8c9005f81e3831d"
SRC_URI[embedding.sha256sum] = "a449c89d80dd9839a42e733db31156d6608dcc5cb70dd33e8dd3991572ddef8e"
SRC_URI[positional.sha256sum] = "d5d46b0fc05fd5d17c5db6e31c79c84a57ff2d787ea6cb2a4dbb3500f26c0765"
SRC_URI[tflite.sha256sum] = "823c11176e8ea1d3ad1c9da2512a6e42431f800780cd1c3df92ba8aa9b11ebbe"
SRC_URI[mel.sha256sum] = "7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c"
SRC_URI[vocab.sha256sum] = "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9"
SRC_URI[sample.sha256sum] = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"

DEMO_DATADIR = "${datadir}/demo-whisper-benchmark"

S = "${WORKDIR}"

inherit allarch

do_configure[noexec] = "1"
do_compile[noexec] = "1"

do_install() {
    install -d ${D}${DEMO_DATADIR}/hailo
    install -m 0644 ${WORKDIR}/tiny-whisper-encoder-10s_15dB.hef ${D}${DEMO_DATADIR}/hailo/
    install -m 0644 ${WORKDIR}/tiny-whisper-decoder-fixed-sequence-matmul-split.hef ${D}${DEMO_DATADIR}/hailo/
    install -m 0644 ${WORKDIR}/token_embedding_weight_tiny.npy ${D}${DEMO_DATADIR}/hailo/
    install -m 0644 ${WORKDIR}/onnx_add_input_tiny.npy ${D}${DEMO_DATADIR}/hailo/

    install -d ${D}${DEMO_DATADIR}/tflite
    install -m 0644 ${WORKDIR}/whisper_tiny_10s_f32.tflite ${D}${DEMO_DATADIR}/tflite/

    install -d ${D}${DEMO_DATADIR}/assets
    install -m 0644 ${WORKDIR}/whisper-mel_filters.npz ${D}${DEMO_DATADIR}/assets/mel_filters.npz
    install -m 0644 ${WORKDIR}/whisper-tiny-vocab.json ${D}${DEMO_DATADIR}/assets/vocab.json
    install -m 0644 ${WORKDIR}/whisper-jfk.wav ${D}${DEMO_DATADIR}/assets/jfk.wav
}

# The Hailo HEFs and the float32 TFLite graph together are ~280 MB, so split
# them out: an image that only demonstrates one accelerator can install just
# the package it needs.
PACKAGES = "${PN}-hailo ${PN}-tflite ${PN}-common ${PN}"

FILES:${PN}-hailo = "${DEMO_DATADIR}/hailo"
FILES:${PN}-tflite = "${DEMO_DATADIR}/tflite"
FILES:${PN}-common = "${DEMO_DATADIR}/assets"

RDEPENDS:${PN}-hailo += "${PN}-common"
RDEPENDS:${PN}-tflite += "${PN}-common"
RDEPENDS:${PN} += "${PN}-hailo ${PN}-tflite ${PN}-common"

ALLOW_EMPTY:${PN} = "1"

INHIBIT_DEFAULT_DEPS = "1"
