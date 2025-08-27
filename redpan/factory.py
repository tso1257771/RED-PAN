#!/usr/bin/env python3
"""
Factory Functions for REDPAN
============================

This module provides convenient factory functions for creating
REDPAN instances with common configurations.
"""

import gc
import logging
from typing import Optional

from .core import REDPAN

logger = logging.getLogger(__name__)


def inference_engine(model,
                 pred_npts: int = 6000,
                 dt: float = 0.01,
                 pred_interval_sec: float = 10.0,
                 batch_size: int = 32,
                 postprocess_config: Optional[dict] = None) -> REDPAN:
    """
    Factory function to create a true SeisBench-style RED-PAN picker
    
    Args:
        model: TensorFlow model for prediction
        pred_npts: Model input length
        dt: Sample rate
        pred_interval_sec: Sliding window step
        batch_size: Batch size for prediction
        postprocess_config: Optional postprocessing configuration
        
    Returns:
        REDPAN instance
    """
    picker = REDPAN(
        model=model,
        pred_npts=pred_npts,
        dt=dt,
        pred_interval_sec=pred_interval_sec,
        batch_size=batch_size
    )
    
    # Add postprocessing config if provided
    if postprocess_config:
        picker.postprocess_config = postprocess_config
    
    # Clean up any temporary objects created during initialization
    gc.collect()
    
    logger.info(f"Created REDPAN engine with batch_size={batch_size}")
    return picker