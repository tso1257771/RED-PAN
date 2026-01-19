"""
REDPAN: High-Performance SeisBench-Inspired RED-PAN Implementation
=========================================================================

A drop-in replacement for RED-PAN's continuous processing (STMF in published paper)
using SeisBench's efficient direct array accumulation approach.

Author: Wu-Yu Liao
Date: July 2025
License: MIT
"""

from .core import REDPAN
from .factory import inference_engine
from .utils import (
    create_gaussian_weights, validate_waveform, 
    sac_len_complement, stream_standardize,
    generate_matching_noise, find_reference_signal,
    pad_waveform_with_noise, fill_flat_regions
)

__version__ = "1.0.0"
__author__ = "Wu-Yu Liao"
__email__ = "tso1257771@gmail.com"

__all__ = [
    "REDPAN",
    "inference_engine", 
    "sac_len_complement",
    "stream_standardize",
    # Noise generation and waveform preprocessing utilities
    "generate_matching_noise",
    "find_reference_signal",
    "pad_waveform_with_noise",
    "fill_flat_regions",
]
