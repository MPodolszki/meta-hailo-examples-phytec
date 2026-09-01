"""Terminal report for the Hailo-8 vs i.MX8MP CPU comparison."""

import json

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


class Style:
    """ANSI helpers that collapse to plain text when colour is off."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def __call__(self, text, *codes):
        if not self.enabled or not codes:
            return text
        return "".join(codes) + text + RESET


def _fmt(value, spec="{:.1f}"):
    if value is None or value != value:  # NaN
        return "-"
    return spec.format(value)


def render(results, audio_info, style):
    lines = []
    add = lines.append

    add("")
    add(style("Whisper-tiny — Hailo-8 vs. i.MX8MP CPU", BOLD))
    add(style(
        f"{audio_info['duration']:.1f} s audio, "
        f"{audio_info['window']:.0f} s window fed to both encoders"
        + (f", {audio_info['speech_offset']:.1f} s leading silence trimmed"
           if audio_info["speech_offset"] else ""),
        DIM,
    ))
    add("")

    # Padding is applied before any colour codes, so escape sequences never
    # count towards a column width.
    columns = [
        ("Backend", 16, "<"),
        ("enc inf/s", 11, ">"),
        ("enc ms", 9, ">"),
        ("dec steps/s", 13, ">"),
        ("dec ms", 9, ">"),
        ("full ms", 10, ">"),
        ("x realtime", 12, ">"),
    ]
    table_width = sum(size for _, size, _ in columns)

    def row(cells):
        return "".join(
            f"{cell:{align}{size}}"
            for cell, (_, size, align) in zip(cells, columns)
        )

    add(style(row([label for label, _, _ in columns]), BOLD))
    add(style("-" * table_width, DIM))

    for result in results:
        if result.error:
            add(f"{result.name:<16}" + style(result.error, RED))
            continue

        realtime = _fmt(result.realtime_factor, "{:.2f}") + "x"
        line = row([
            result.name,
            _fmt(result.encoder.per_second, "{:.2f}"),
            _fmt(result.encoder.median * 1000, "{:.0f}"),
            _fmt(result.decoder.per_second, "{:.2f}"),
            _fmt(result.decoder.median * 1000, "{:.0f}"),
            _fmt(result.transcription.median * 1000, "{:.0f}"),
            realtime,
        ])
        # Colourise the realtime column only, after it has been padded.
        padded = f"{realtime:>{columns[-1][1]}}"
        colour = GREEN if result.realtime_factor >= 1.0 else YELLOW
        add(line[:-len(padded)] + style(padded, colour))

    speedups = _speedups(results, style)
    if speedups:
        add("")
        add(speedups)
    add("")

    add(style("Transcripts", BOLD))
    for result in results:
        if result.error:
            continue
        text = result.transcript.strip() or "(empty)"
        add(f"  {result.name:<16} {text}")
        add(style(
            f"  {'':<16} {result.tokens} tokens, stopped on {result.stop_reason}, "
            f"decoder sequence {result.details.get('sequence_length', '?')}",
            DIM,
        ))
    add("")

    add(style("Setup", BOLD))
    for result in results:
        if result.error:
            continue
        details = result.details
        add(f"  {result.name:<16} {details.get('runtime')} / "
            f"{details.get('delegate')}")
        add(style(f"  {'':<16} {details.get('model')}", DIM))
        add(style(
            f"  {'':<16} load {result.load_seconds:.1f} s, "
            f"first inference {result.warmup_seconds:.1f} s",
            DIM,
        ))
    add("")

    return "\n".join(lines)


def _speedups(results, style):
    """Compare every backend against the slowest one on end-to-end latency."""
    usable = [r for r in results if not r.error and r.transcription.median > 0]
    if len(usable) < 2:
        return ""

    baseline = max(usable, key=lambda r: r.transcription.median)
    parts = []
    for result in usable:
        if result is baseline:
            continue
        factor = baseline.transcription.median / result.transcription.median
        parts.append(f"{result.name} {style(f'{factor:.1f}x', BOLD)}")

    return style(f"Speed-up vs. {baseline.name}: ", DIM) + ", ".join(parts)


def to_json(results, audio_info):
    return json.dumps(
        {
            "audio": audio_info,
            "results": [result.as_dict() for result in results],
        },
        indent=2,
    )
