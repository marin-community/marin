# GB200 streaming-attention backward replay blocked

The bounded batch-priority reservation reported one GB200. Device verification inside the allocated runtime exposed an NVIDIA H100 80GB HBM3 with compute capability 9.0 and UUID `GPU-7cf3cc97-9a2b-6f82-aaa0-35a9b1d41f0e`, which is the H100 used by the separate H100 replay.

The hardware mismatch was detected before dependency installation, compilation, or benchmark execution. The reservation was released and cleanup was verified. No GB200 performance or compatibility conclusion can be drawn from this attempt.
