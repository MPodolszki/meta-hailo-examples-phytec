SUMMARY = "Play and record sound with Python, via PortAudio"
HOMEPAGE = "https://python-sounddevice.readthedocs.io/"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=3bd2f5dae3009c67198af44e33402296"

# Pinned below 0.5.2: newer releases declare license = "MIT" as a bare PEP 639
# string in pyproject.toml, which the setuptools version in this release's
# native sysroot fails to validate ("must be valid exactly by one
# definition"). 0.5.1 still uses the classic setup.py license='MIT' keyword.
SRC_URI[sha256sum] = "09ca991daeda8ce4be9ac91e15a9a81c8f81efa6b695a348c9171ea0c16cb041"

inherit pypi setuptools3

# sounddevice_build.py calls cffi's ffibuilder.set_source(..., None) --
# ABI-mode, out-of-line: this generates a plain _sounddevice.py that
# dlopen()s libportaudio at import time, no C compiler needed at build time.
DEPENDS += "python3-cffi-native"

RDEPENDS:${PN}:class-target = " \
    portaudio-v19 \
    python3-cffi \
    python3-ctypes \
    python3-numpy \
"

BBCLASSEXTEND = "native nativesdk"
