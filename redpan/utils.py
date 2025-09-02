#!/usr/bin/env python3
"""
Utility Functions for REDPAN
=============================

This module contains utility functions for weight calculation,
waveform validation, and other helper functions.
"""

import gc
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_gaussian_weights(pred_npts: int, sigma_factor: float = 6.0) -> np.ndarray:
    """
    Create Gaussian position weights exactly like SeisBench
    
    These weights give higher importance to predictions from the center
    of each window, which are generally more reliable than edge predictions.
    
    Args:
        pred_npts: Length of prediction window
        sigma_factor: Factor to determine sigma (pred_npts / sigma_factor)
        
    Returns:
        Gaussian weights centered on prediction window
    """
    center = pred_npts // 2
    sigma = pred_npts / sigma_factor  # SeisBench default: 6.0
    positions = np.arange(pred_npts)
    weights = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
    
    # Normalize to prevent weight inflation
    weights = weights / np.mean(weights)
    
    # Clean up intermediate variables
    del positions
    gc.collect()
    
    logger.debug(f"Created Gaussian weights: center={center}, sigma={sigma:.1f}, "
                f"range=[{weights.min():.3f}, {weights.max():.3f}]")
    
    return weights.astype(np.float32)


def validate_waveform(wf, min_length: int) -> None:
    """
    Validate input waveform for processing
    
    Args:
        wf: Input waveform (Stream or numpy array)
        min_length: Minimum required length
        
    Raises:
        ValueError: If waveform is invalid
    """
    if wf is None:
        raise ValueError("Waveform cannot be None")
    
    # Handle numpy arrays
    if isinstance(wf, np.ndarray):
        if len(wf) == 0:
            raise ValueError("Waveform array is empty")
        if len(wf) < min_length:
            raise ValueError(f"Data should be longer than {min_length} points, got {len(wf)}")
        return
    
    # Handle ObsPy Streams
    if hasattr(wf, '__len__'):
        if len(wf) == 0:
            raise ValueError("Waveform stream is empty")
        
        if hasattr(wf[0], 'data'):
            if len(wf[0].data) < min_length:
                raise ValueError(f"Data should be longer than {min_length} points, "
                               f"got {len(wf[0].data)}")
            
            # Check for consistent sampling across components
            if len(wf) > 1:
                lengths = [len(trace.data) for trace in wf]
                if len(set(lengths)) > 1:
                    logger.warning(f"Inconsistent trace lengths: {lengths}")
            return
        else:
            raise ValueError("Invalid Stream format - traces do not have 'data' attribute")
    else:
        raise ValueError(f"Invalid input type: {type(wf)}. Expected numpy array or ObsPy Stream")


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights to prevent accumulation bias
    
    Args:
        weights: Input weights array
        
    Returns:
        Normalized weights
    """
    if weights.sum() == 0:
        return weights
    
    return weights / np.mean(weights)


def create_triangular_weights(pred_npts: int) -> np.ndarray:
    """
    Create triangular position weights as alternative to Gaussian
    
    Args:
        pred_npts: Length of prediction window
        
    Returns:
        Triangular weights centered on prediction window
    """
    center = pred_npts // 2
    positions = np.arange(pred_npts)
    weights = 1 - np.abs(positions - center) / center
    weights = np.maximum(weights, 0.1)  # Minimum weight threshold
    
    # Clean up intermediate variables
    del positions
    gc.collect()
    
    return normalize_weights(weights).astype(np.float32)


def create_cosine_weights(pred_npts: int) -> np.ndarray:
    """
    Create cosine position weights as alternative to Gaussian
    
    Args:
        pred_npts: Length of prediction window
        
    Returns:
        Cosine weights centered on prediction window  
    """
    positions = np.arange(pred_npts)
    weights = 0.5 * (1 + np.cos(np.pi * (positions - pred_npts/2) / (pred_npts/2)))
    
    # Clean up intermediate variables
    del positions
    gc.collect()
    
    return normalize_weights(weights).astype(np.float32)

def align_wf_starttime(wf, target_starttime):
    '''Pad the waveform if start time is later than target_starttime'''
    for trc in wf:
        if trc.stats.starttime > target_starttime:
            time_diff = trc.stats.starttime - target_starttime
            # create padding array
            pad_samples = int(np.ceil(time_diff / trc.stats.delta))
            pad_value = np.median(trc.data)
            padding = np.full(pad_samples, pad_value)
            # concatenate padding and original data
            trc.data = np.concatenate([padding, trc.data])
            # update starttime
            trc.stats.starttime = target_starttime
    return wf

def sac_len_complement(wf, max_length=None):
    '''Complement sac data into the same length
    '''

    if not wf or len(wf) == 0:
        return wf
    
    # Determine target length efficiently
    if max_length is None:
        max_length = max((len(trace.data) for trace in wf), default=0)
        if max_length == 0:
            return wf
    
    # Process traces in-place for memory efficiency
    for trace in wf:
        current_len = len(trace.data)
        
        if current_len == max_length:
            continue
        elif current_len == 0:
            trace.data = np.zeros(max_length, dtype=trace.data.dtype)
        elif current_len < max_length:
            # Use np.resize for efficiency (handles padding automatically)
            last_val = trace.data[-1] if current_len > 0 else 0
            trace.data = np.concatenate([
                trace.data, 
                np.full(max_length - current_len, last_val, 
                        dtype=trace.data.dtype)
            ])
        else:
            trace.data = trace.data[:max_length]
    
    # Clean up any temporary arrays
    gc.collect()
    return wf


def stream_standardize(st, data_length):
    """
    input: obspy.stream object (raw data)
    output: obspy.stream object (standardized)
    """
    data_len = [len(i.data) for i in st]
    check_len = np.array_equal(data_len, np.repeat(data_length, 3))
    if not check_len:
        st = sac_len_complement(st, max_length=data_length)

    st = st.detrend("demean")
    for s in st:
        data_valid = s.data[~np.isnan(s.data) & ~np.isinf(s.data)]
        data_std = np.std(data_valid)
        if data_std == 0:
            data_std = 1
        s.data /= data_std
        s.data[np.isinf(s.data)] = data_valid.mean()
        s.data[np.isnan(s.data)] = data_valid.mean()
    return st
