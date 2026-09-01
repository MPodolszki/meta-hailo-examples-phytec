SUMMARY = "Model and assets for the YOLOv8n object detection demo"
DESCRIPTION = "Collects everything the object-detection demo needs at \
               runtime: the YOLOv8n HEF (Hailo Model Zoo v2.17.0, compiled \
               for Hailo-8 with HailoRT on-chip NMS postprocessing built in), \
               the same YOLOv8n graph as an uncompiled ONNX model for the \
               i.MX8MP CPU backend, the COCO-80 class labels, and a \
               reference sample image."
HOMEPAGE = "https://www.phytec.de"

# The HEF and labels are redistributed from Hailo's Model Zoo / example repo
# (MIT); the sample image is Hailo's own "dog and bicycle" demo image. The
# uncompiled ONNX graph is the YOLOv8n model itself (Ultralytics, AGPL-3.0),
# from the same Model Zoo source the HEF was compiled from (see
# hailo_model_zoo's cfg/networks/yolov8n.yaml on-device).
LICENSE = "MIT & AGPL-3.0-only"
LIC_FILES_CHKSUM = " \
    file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302 \
    file://${COMMON_LICENSE_DIR}/AGPL-3.0-only;md5=73f1eb20517c55bf9493b7dd6e480788 \
"

HAILO_MODELZOO = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
HAILO_RESOURCES = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources"

# YOLOv8n for the Hailo-8, Model Zoo v2.17.0 - the release that matches
# HailoRT 4.23.0 (the 'hailo8' branch line this image builds, see
# phytec-hailo-image.bb / hailo8-python-wheels). The HEF has HailoRT's
# on-chip NMS postprocessing built in, so the host only has to letterbox the
# input and draw the already-decoded boxes (see object_detect/postprocess.py).
SRC_URI = " \
    ${HAILO_MODELZOO}/v2.17.0/hailo8/yolov8n.hef;name=hef \
"

# The uncompiled YOLOv8n ONNX graph the HEF above was compiled from (Model
# Zoo v2.17.0's yolov8n.yaml network_path) -- same weights, but without
# HailoRT's on-chip NMS postprocessing layer, so decode+NMS happen on the
# host for the CPU backend (see object_detect/postprocess.py). The zip also
# holds yolov8n_nms_config.json (the NMS thresholds this graph was compiled
# with); only the .onnx is used here.
SRC_URI += " \
    https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8n/2023-01-30/yolov8n.zip;name=onnx \
"

# COCO-80 class names in the order Hailo's object_detection example ships
# them, and a reference sample (Hailo's own "dog and bicycle" demo image,
# used by that same example).
SRC_URI += " \
    https://raw.githubusercontent.com/hailo-ai/Hailo-Application-Code-Examples/${HAEX_REV}/runtime/python/common/coco.txt;name=labels;downloadfilename=object-detection-labels.txt \
    ${HAILO_RESOURCES}/images/dog_bicycle.jpg;name=sample;downloadfilename=object-detection-sample.jpg \
"

HAEX_REV = "7b850e0444a142ada88443d675e3c3ddda18bd74"

SRC_URI[hef.sha256sum] = "e893b0f9dcae366fe1bc9ebce25e32ad889acf2bc58cfe1f73a572f78f7ec055"
SRC_URI[onnx.sha256sum] = "478beae8c59e7fa0a808c2ca027fe32bd07dbaf2f1abbb7f670957d512f5f6fd"
SRC_URI[labels.sha256sum] = "d7654b26101572841ed1cd80aa03aa60e35f1b8acb4aea6906c4066886f16e07"
SRC_URI[sample.sha256sum] = "06089519900e7cdf28963d049c8ead5615cd86905ef2b08cfb6bd2eb120ae654"

DEMO_DATADIR = "${datadir}/demo-object-detection"

S = "${WORKDIR}"

inherit allarch

do_configure[noexec] = "1"
do_compile[noexec] = "1"

do_install() {
    install -d ${D}${DEMO_DATADIR}/hailo
    install -m 0644 ${WORKDIR}/yolov8n.hef ${D}${DEMO_DATADIR}/hailo/

    install -d ${D}${DEMO_DATADIR}/cpu
    install -m 0644 ${WORKDIR}/yolov8n.onnx ${D}${DEMO_DATADIR}/cpu/

    install -d ${D}${DEMO_DATADIR}/assets
    install -m 0644 ${WORKDIR}/object-detection-labels.txt ${D}${DEMO_DATADIR}/assets/labels.txt
    install -m 0644 ${WORKDIR}/object-detection-sample.jpg ${D}${DEMO_DATADIR}/assets/dog_bicycle.jpg
}

FILES:${PN} = "${DEMO_DATADIR}"

ALLOW_EMPTY:${PN} = "1"

INHIBIT_DEFAULT_DEPS = "1"

COMPATIBLE_MACHINE = "(mx8mp-nxp-bsp)"
