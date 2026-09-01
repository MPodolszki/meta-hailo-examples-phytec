# demo-loop

Unattended kiosk rotation for a customer-facing board: cycles Facematch,
Object Detection (Hailo-8 vs. CPU compare mode) and Whisper speech-to-text,
30 seconds each, forever, plus a static 4th "more demos" info screen.

- `detect-display` runs once at boot, before Weston starts, and picks
  whichever of HDMI/LVDS is actually connected (HDMI preferred), writing the
  choice into Weston's config and into `/run/demo-display-kind` for the
  other pieces here to read.
- `demo-orchestrator` is the actual loop: it launches each demo's real
  command, waits for it to signal `/run/demo-loop/ready` (so loading time
  isn't stolen from the 30s a visitor gets to look at the result), then
  gives it 30s before moving on.
- `object-detection-viewer` is a small GTK viewer for the JPEG frame that
  `object-detection-*-cam`/`-video`/`-image` write (those commands
  deliberately don't open a window themselves, see their own --help).
  `_od-with-viewer` pairs any one of those commands with this viewer for
  someone running it by hand; this package's postinst renames the real
  binaries to `*.real` and installs `_od-with-viewer`-based wrapper scripts
  at their original names.
- `whisper-demo-gui` is a small GTK front-end around whisper_bench that
  transcribes the bundled sample audio and plays it back.
- `demo-info-screen` is the static 4th slide, listing every other demo
  command that stays manually runnable (over the board's Debug UART) and
  how to stop the loop.

Stop the loop: `systemctl stop demo-loop.service`.
