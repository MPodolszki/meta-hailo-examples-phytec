# demo-loop's orchestrator waits for a demo to touch /run/demo-loop/ready
# before it starts that slot's 30s countdown, so "loading the model" is
# visible time rather than time stolen from the visitor. object-detection,
# whisper and the info screen are demo-loop's own binaries and signal it
# directly; demo-celebrity-face-match is upstream PHYTEC code that knows
# nothing about the loop, so without this it never signals and every one of
# its slots burns the orchestrator's full MAX_LOADING_SECONDS=90 bounded
# wait first. Measured on phyBOARD-Pollux i.MX8MP: it is actually ready
# after ~20.7s, so the loop was wasting ~69s per rotation on a loading
# screen nobody needed to look at.
#
# Done here, in demo-loop's own layer, rather than upstream: the ready-file
# protocol is demo-loop's invention and the upstream demo must stay usable
# stand-alone (touch_ready() is a no-op when /run/demo-loop does not exist,
# which is the case unless demo-loop.service -- which declares
# RuntimeDirectory=demo-loop -- is running).
#
# Anchored source edits rather than a .patch on purpose: a context diff
# silently rots into a fuzzy or failed apply on an SRCREV bump, whereas
# each assert below names exactly what it could no longer find.

python do_patch:append() {
    import os

    src = None
    for root, dirs, files in os.walk(d.getVar("S")):
        if "aidemo.py" in files:
            src = os.path.join(root, "aidemo.py")
            break
    if src is None:
        bb.fatal("demo-celebrity-face-match: aidemo.py not found under S; "
                 "cannot add the demo-loop ready signal")

    with open(src, encoding="utf-8") as fd:
        text = fd.read()

    if "touch_ready" in text:
        bb.note("demo-celebrity-face-match: ready signal already present, skipping")
        return

    edits = [
        # 1. pathlib, matching how the other three demos touch the file.
        ("import argparse\nimport os\nimport sys\nimport time\n",
         "import argparse\nimport os\nimport pathlib\nimport sys\nimport time\n"),
        # 2. the helper, before the first module-level constant.
        ('FRAME_HEIGHT = {"hdmi": 800, "lvds": 600}',
         'READY_FILE = "/run/demo-loop/ready"\n'
         '\n'
         '\n'
         'def touch_ready():\n'
         '    """Tells demo-orchestrator "the loading screen is done, start the 30s\n'
         '    countdown now" -- see its module docstring. A no-op when this demo is\n'
         '    run by hand: /run/demo-loop only exists while demo-loop.service (which\n'
         '    declares RuntimeDirectory=demo-loop) is up."""\n'
         '    try:\n'
         '        pathlib.Path(READY_FILE).touch()\n'
         '    except OSError:\n'
         '        pass\n'
         '\n'
         '\n'
         'FRAME_HEIGHT = {"hdmi": 800, "lvds": 600}'),
        # 3. the call site: load_ai() sets loaded_event exactly once, right
        #    after it destroys the load screen and shows the real window --
        #    that is precisely the moment the orchestrator wants to hear
        #    about. The early "Failed to open video device" return path
        #    deliberately does not signal, so a camera-less board still
        #    falls back to the bounded wait instead of showing an error
        #    screen for a full slot.
        ("        self.loaded_event.set()\n",
         "        self.loaded_event.set()\n        touch_ready()\n"),
    ]

    for anchor, replacement in edits:
        found = text.count(anchor)
        if found != 1:
            bb.fatal("demo-celebrity-face-match: expected exactly 1 occurrence of "
                     "%r in aidemo.py, found %d -- upstream changed, "
                     "re-check the demo-loop ready signal" % (anchor[:60], found))
        text = text.replace(anchor, replacement)

    with open(src, "w", encoding="utf-8") as fd:
        fd.write(text)

    bb.note("demo-celebrity-face-match: added demo-loop ready signal to %s" % src)
}
