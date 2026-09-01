# Wrap openat2 so pseudo retains directory file descriptor tracking on newer hosts.
FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRCREV = "43cbd8fb4914328094ccdb4bb827d74b1bac2046"
PV = "1.9.3+git"
SRC_URI:remove = "file://0001-configure-Prune-PIE-flags.patch"