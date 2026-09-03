SUMMARY = "Unattended kiosk rotation: Facematch / Object Detection / Whisper"
DESCRIPTION = "Cycles demo-celebrity-face-match, object-detection-compare-cam \
               (Hailo-8 vs. CPU) and a Whisper-tiny sample transcription, 30s \
               each, forever, plus a static info screen pointing at every \
               other demo command that stays manually runnable. Also detects \
               once at boot whether HDMI or LVDS is connected and configures \
               Weston accordingly. See README.md."
HOMEPAGE = "https://www.phytec.de"

LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://detect-display \
    file://demo-orchestrator \
    file://object-detection-viewer \
    file://_od-with-viewer \
    file://whisper-demo-gui \
    file://demo-info-screen \
    file://demo-loop.service \
    file://weston-output-config.service \
    file://output-detect.conf \
    file://99-hailo.rules \
    file://object-detection-demo.conf \
    file://50-demo-loop-overrides.preset \
    file://README.md \
"

S = "${WORKDIR}"

inherit systemd

SYSTEMD_SERVICE:${PN} = "demo-loop.service weston-output-config.service"
SYSTEMD_AUTO_ENABLE:${PN} = "enable"

# The object-detection-* commands this wraps (see WRAPPED_OD_COMMANDS below)
# are demo-object-detection's; renaming them here rather than there keeps
# that recipe usable stand-alone (its own README already tells someone
# running it by hand that it writes a JPEG file with no window of its own).
WRAPPED_OD_COMMANDS = " \
    object-detection-image \
    object-detection-camera \
    object-detection-hailo-cam \
    object-detection-cpu-cam \
    object-detection-hailo-video \
    object-detection-cpu-video \
    object-detection-compare-cam \
    object-detection-compare-video \
"

do_configure[noexec] = "1"
do_compile[noexec] = "1"

do_install() {
    install -d ${D}${bindir}
    for cmd in detect-display demo-orchestrator object-detection-viewer \
               _od-with-viewer whisper-demo-gui demo-info-screen; do
        install -m 0755 ${WORKDIR}/$cmd ${D}${bindir}/$cmd
    done

    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/demo-loop.service ${D}${systemd_system_unitdir}/
    install -m 0644 ${WORKDIR}/weston-output-config.service ${D}${systemd_system_unitdir}/

    install -d ${D}${systemd_system_unitdir}/weston.service.d
    install -m 0644 ${WORKDIR}/output-detect.conf \
        ${D}${systemd_system_unitdir}/weston.service.d/

    install -d ${D}${sysconfdir}/udev/rules.d
    install -m 0644 ${WORKDIR}/99-hailo.rules ${D}${sysconfdir}/udev/rules.d/

    install -d ${D}${sysconfdir}/tmpfiles.d
    install -m 0644 ${WORKDIR}/object-detection-demo.conf \
        ${D}${sysconfdir}/tmpfiles.d/

    install -d ${D}${systemd_unitdir}/system-preset
    install -m 0644 ${WORKDIR}/50-demo-loop-overrides.preset \
        ${D}${systemd_unitdir}/system-preset/

    install -d ${D}${docdir}/${BPN}
    install -m 0644 ${WORKDIR}/README.md ${D}${docdir}/${BPN}/
}

FILES:${PN} = " \
    ${bindir}/detect-display \
    ${bindir}/demo-orchestrator \
    ${bindir}/object-detection-viewer \
    ${bindir}/_od-with-viewer \
    ${bindir}/whisper-demo-gui \
    ${bindir}/demo-info-screen \
    ${systemd_system_unitdir}/demo-loop.service \
    ${systemd_system_unitdir}/weston-output-config.service \
    ${systemd_system_unitdir}/weston.service.d/output-detect.conf \
    ${sysconfdir}/udev/rules.d/99-hailo.rules \
    ${sysconfdir}/tmpfiles.d/object-detection-demo.conf \
    ${systemd_unitdir}/system-preset/50-demo-loop-overrides.preset \
    ${docdir}/${BPN} \
"

# demo-celebrity-face-match ships its own always-on service; the loop takes
# over as the sole autostart, so that one gets disabled -- primarily by the
# 50-demo-loop-overrides.preset installed above (which beats that recipe's
# own 98-*.preset no matter what order the postinsts run in), with the
# pkg_postinst below as a second net for upgrades on a live system.
# Leaving it enabled is not cosmetic: it holds /dev/video0 for good and
# every camera slot the loop then starts fails on a busy sensor.
# object-detection-*/whisper-* are pure CLI tools with no autostart of
# their own, nothing to disable there.
RDEPENDS:${PN} += " \
    demo-celebrity-face-match \
    demo-object-detection \
    demo-whisper-benchmark \
    python3-core \
    python3-pygobject \
    gtk+3 \
    alsa-utils \
    pulseaudio-utils \
"

pkg_postinst:${PN}() {
#!/bin/sh -e
BINDIR="$D${bindir}"

for cmd in ${WRAPPED_OD_COMMANDS}; do
    plain="$BINDIR/$cmd"
    real="$BINDIR/$cmd.real"
    if [ -e "$plain" ] && [ ! -e "$real" ]; then
        mv "$plain" "$real"
        cat > "$plain" <<WRAPPER
#!/bin/bash
exec ${bindir}/_od-with-viewer "${bindir}/$cmd.real" "\$@"
WRAPPER
        chmod 0755 "$plain"
    fi
done

if [ -n "$D" ]; then
    systemctl --root="$D" disable demo-celebrity-face-match.service || true
else
    systemctl disable demo-celebrity-face-match.service || true
fi
}

COMPATIBLE_MACHINE = "(mx8mp-nxp-bsp)"
