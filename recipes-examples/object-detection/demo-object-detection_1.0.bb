SUMMARY = "Object detection: YOLOv8n on the Hailo-8 AI accelerator vs the i.MX8MP CPU"
DESCRIPTION = "Runs YOLOv8n on the Hailo-8 (Hailo Model Zoo, HailoRT on-chip \
               NMS postprocessing built into the HEF) and, as a comparison, \
               the same uncompiled YOLOv8n graph on the i.MX8MP Cortex-A53 \
               cores via OpenCV's DNN module, and draws the resulting \
               bounding boxes. Installs object-detection-image (single image, \
               a bundled sample by default), object-detection-camera (a live \
               camera feed or a video file) and object-detection-benchmark \
               (timed latency/FPS for both backends on the same image), each \
               taking -b/--backend {hailo,cpu} (default: hailo; -camera and \
               -benchmark also take 'compare', to run every backend on the \
               same input side by side). Also installs six fixed-backend/ \
               fixed-source convenience wrappers around object-detection-camera \
               that need no flags: object-detection-{hailo,cpu,compare}-cam \
               (camera, default index 0) and object-detection-{hailo,cpu,compare}-video \
               (a video file, given as the first argument). Detection only, \
               no tracking -- see README.md."
HOMEPAGE = "https://www.phytec.de"

LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

# The three flag-driven commands, plus the fixed-backend/fixed-source
# convenience wrappers around object-detection-camera (see camera_cli.run()).
OD_COMMANDS = " \
    object-detection-image \
    object-detection-camera \
    object-detection-benchmark \
    object-detection-hailo-cam \
    object-detection-hailo-video \
    object-detection-cpu-cam \
    object-detection-cpu-video \
    object-detection-compare-cam \
    object-detection-compare-video \
"

SRC_URI = " \
    file://object_detect \
    file://object-detection-image \
    file://object-detection-camera \
    file://object-detection-benchmark \
    file://object-detection-hailo-cam \
    file://object-detection-hailo-video \
    file://object-detection-cpu-cam \
    file://object-detection-cpu-video \
    file://object-detection-compare-cam \
    file://object-detection-compare-video \
    file://README.md \
"

S = "${WORKDIR}"

inherit python3-dir

do_configure[noexec] = "1"
do_compile[noexec] = "1"

do_install() {
    install -d ${D}${PYTHON_SITEPACKAGES_DIR}/object_detect
    install -m 0644 ${WORKDIR}/object_detect/*.py ${D}${PYTHON_SITEPACKAGES_DIR}/object_detect/

    install -d ${D}${bindir}
    for cmd in ${OD_COMMANDS}; do
        install -m 0755 ${WORKDIR}/$cmd ${D}${bindir}/$cmd
    done

    install -d ${D}${docdir}/${BPN}
    install -m 0644 ${WORKDIR}/README.md ${D}${docdir}/${BPN}/
}

FILES:${PN} = " \
    ${bindir}/object-detection-image \
    ${bindir}/object-detection-camera \
    ${bindir}/object-detection-benchmark \
    ${bindir}/object-detection-hailo-cam \
    ${bindir}/object-detection-hailo-video \
    ${bindir}/object-detection-cpu-cam \
    ${bindir}/object-detection-cpu-video \
    ${bindir}/object-detection-compare-cam \
    ${bindir}/object-detection-compare-video \
    ${PYTHON_SITEPACKAGES_DIR}/object_detect \
    ${docdir}/${BPN} \
"

# The CPU backend degrades gracefully on its own (cpu_infer.py only imports
# cv2, already required below), but yolov8n.hef is compiled specifically for
# Hailo-8, so this package as a whole still always needs the Hailo-8 Python
# bindings - it's only pulled into phytec-hailo-image.bb when
# HAILO_CHIP = "hailo8" (the default).
RDEPENDS:${PN} += " \
    demo-object-detection-data \
    hailo8-python-wheels \
    python3-core \
    python3-numpy \
    python3-opencv \
"

RRECOMMENDS:${PN} += "hailo8-firmware libhailort hailo8-pci"

COMPATIBLE_MACHINE = "(mx8mp-nxp-bsp)"
