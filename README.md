# meta-hailo-examples-phytec

Demo applications for the **PHYTEC Hailo AI Kit** (phyBOARD-Pollux i.MX8MP +
Hailo-8). This layer holds everything customer-facing that runs on the board;
the BSP integration — HailoRT, firmware, PCIe driver, the image recipe — lives
in [meta-hailo-phytec](https://github.com/MPodolszki/meta-hailo-phytec), whose
README covers building and flashing the kit.

Both layers are activated by the kit manifest,
`BSP-Yocto-HailoAIKit-i.MX8MP-v0.1.xml`. There is no `bblayers.conf` to edit —
it is generated from the manifest.

| | |
|---|---|
| Layer | `hailo-examples-phytec`, `BBFILE_PRIORITY` 8 |
| `LAYERDEPENDS` | `core` |
| `LAYERSERIES_COMPAT` | `scarthgap` |

`LAYERDEPENDS` names only `core`, although the recipes here `RDEPENDS` on
`demo-celebrity-face-match` and bbappend it. That dependency is satisfied in
practice because the kit manifest always activates both layers together.

---

## What is in here

### `recipes-examples/demo-loop` — the kiosk

`demo-loop.service` is the only demo that autostarts on the shipped image. It
rotates four slots, 30 s each, forever, on whichever display was found at boot.

| Piece | Role |
|---|---|
| `detect-display` | Runs before Weston (`weston-output-config.service`). Picks HDMI or LVDS — HDMI wins — rewrites the `[output]` stanzas in `/etc/xdg/weston/weston.ini`, disables the other head explicitly, and records the choice in `/run/demo-display-kind`. No hotplug: connect the display before powering on. |
| `demo-orchestrator` | The loop itself. Launches each demo, waits for it to touch `/run/demo-loop/ready`, then starts that slot's 30 s. |
| `object-detection-viewer` | GTK viewer for the JPEG frames the `object-detection-*` commands write. `_od-with-viewer` pairs the two for interactive use; the postinst renames the real binaries to `*.real` and installs wrappers under the original names. |
| `whisper-demo-gui` | GTK front-end around `whisper_bench`: transcribes the bundled sample and plays it back through `paplay`. |
| `demo-info-screen` | Static 4th slide listing the demos that stay runnable by hand. |

Stop it with `systemctl stop demo-loop.service`.

**Why the ready signal exists.** Model loading is slow — face match alone takes
~20 s. Without a ready signal the orchestrator could only guess, and a fixed
wait either cuts into the visitor's 30 s or wastes it. Each demo therefore
touches `/run/demo-loop/ready` once it has something interactive on screen, and
the loading screen it draws itself is visible time rather than stolen time. A
demo that never signals still gets its 30 s after a bounded wait, so a missing
signal cannot stall the loop.

### `recipes-examples/celebrity-face-match` — bbappend only

`demo-celebrity-face-match_%.bbappend` does two things to the upstream demo,
without forking it:

1. **Adds the ready signal.** Three anchored edits in `do_patch:append()` —
   `import pathlib`, a module-level `READY_FILE`/`touch_ready()` matching the
   other demos, and the call after `loaded_event.set()`. Anchored edits with
   `bb.fatal` asserts rather than a `.patch` on purpose: a context diff rots
   silently on an `SRCREV` bump, whereas each assert names exactly what it
   could no longer find. `touch_ready()` is a no-op when run standalone
   (`/run/demo-loop` only exists while the loop runs), so the upstream demo
   stays usable by hand.

2. **Disables its autostart** via `SYSTEMD_AUTO_ENABLE:${PN} = "disable"`.
   demo-loop is meant to be the sole autostart — it launches face match itself
   as one of its slots. Left enabled, the standalone service starts at boot and
   holds `/dev/video0` forever; every camera slot the loop then starts dies with
   `mipi_csis_set_fmt, set sensor format fail` in dmesg.

   This has to be `SYSTEMD_AUTO_ENABLE` and not a preset file. With it set to
   `"enable"`, `systemd.bbclass` runs a plain `systemctl --root=$D enable`
   during rootfs assembly, which creates the `multi-user.target.wants` symlink
   unconditionally and never consults presets at all — the bbclass only calls
   `systemctl preset` in the `$D`-is-empty branch, i.e. on a live system.
   demo-loop's `50-demo-loop-overrides.preset` and its `pkg_postinst` are kept
   as second nets for upgrades on a running board.

### `recipes-examples/object-detection`

YOLOv8n on the Hailo-8 vs. the i.MX8MP CPU, with on-chip NMS.
`demo-object-detection-data` carries the model and assets (~5 MB). The
`object-detection-*` commands deliberately do not open a window themselves —
they write JPEG frames, and `object-detection-viewer` from demo-loop displays
them. See each command's `--help`.

### `recipes-examples/whisper-benchmark`

Whisper speech-to-text, Hailo-8 vs. i.MX8MP CPU. Microphone capture goes
through `python3-sounddevice` → PortAudio. `recipes-support/portaudio` pins
that to the **ALSA backend only**: upstream also builds against JACK, which
makes `libportaudio.so.2` `dlopen()` a full JACK server for a demo that never
uses it.

Playback is a different path and does need PulseAudio: `whisper-demo-gui` calls
`paplay` and only falls back to `aplay`, because PulseAudio autospawns in this
session and holds the ALSA device exclusively — a plain `aplay` would hit
`Device or resource busy` and the slot would run silently. `paplay` comes from
`pulseaudio-misc` (there is no `pulseaudio-utils` package in OpenEmbedded —
that is the Debian name) and the daemon from `pulseaudio-server`; both are
named explicitly in demo-loop's `RDEPENDS` rather than borrowed from
`packagegroup-bluetooth`, which is where they happen to come from today.

Without an external USB speaker there is nothing to play through — the SoC's
built-in `audiohdmi` card alone does not count, and the demo says so on screen.

---

## Running the demos by hand

Stop the loop first, then over the debug UART or SSH:

```sh
object-detection-image [-b hailo|cpu]      # single image
object-detection-hailo-video <file>        # video file, Hailo-8
object-detection-cpu-video <file>          # video file, CPU
object-detection-compare-cam               # live camera, Hailo-8 vs CPU
object-detection-benchmark                 # text-only speed comparison
whisper-hailo -m                           # microphone, Hailo-8
whisper-cpu [file.wav]                     # CPU instead of Hailo-8
whisper-benchmark                          # text-only speed comparison
```

---

## Runtime paths

| Path | Written by | Purpose |
|---|---|---|
| `/run/demo-display-kind` | `detect-display` | `hdmi` or `lvds`, read by the orchestrator and by face match's `-s` flag |
| `/run/demo-loop/ready` | each demo | ready signal; the directory comes from `RuntimeDirectory=demo-loop`, owned by the `weston` user the service runs as |
| `/run/object-detection-demo/camera.jpg` | `object_detect.camera_cli` | current frame for the viewer. Mode 0777 **without** the sticky bit on purpose (`/etc/tmpfiles.d/object-detection-demo.conf`): root and the `weston` user must be able to replace each other's leftovers. A `/tmp` path bit us here once — sticky means only the owner may replace a file, so a root test run over SSH left a file the weston-run loop could never overwrite again. |

---

## License

MIT.
