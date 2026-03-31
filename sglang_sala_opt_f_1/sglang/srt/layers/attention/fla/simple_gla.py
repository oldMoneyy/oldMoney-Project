# SimpleGLA ops: use fla package for chunk (prefill), vendored for recurrent (decode)
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

# Use fla package chunk kernels (known correct)
from fla.ops.simple_gla import chunk_simple_gla

# Use vendored recurrent kernel (optimized for decode)
from sglang.srt.layers.attention.fla.simple_gla_recurrent import fused_recurrent_simple_gla
