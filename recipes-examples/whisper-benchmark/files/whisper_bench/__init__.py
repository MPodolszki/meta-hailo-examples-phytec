"""Whisper-tiny throughput comparison: Hailo-8 vs. i.MX8MP CPU.

Both backends run the same Whisper-tiny model over the same 10 s audio window,
so encoder inferences per second and decoder steps per second are directly
comparable numbers. There is no i.MX8MP NPU backend: measured, it is slower
than the CPU on this model (see README.md), so shipping it would suggest the
NPU is a viable path for Whisper-sized transformers, which it is not.
"""

__version__ = "1.0"
