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


def sac_len_complement(wf, max_length=None, pad_mode='noise'):
    """
    Complement SAC data into the same length with configurable padding.
    
    Args:
        wf: ObsPy Stream object
        max_length: Target length (if None, uses max trace length)
        pad_mode: Padding method for shorter traces
            - 'noise': Spectrum-matched background noise (default)
            - 'repeat_last': Repeat last sample value (original behavior)
            - 'zero': Zero padding
            
    Returns:
        Stream with all traces padded to max_length
    """

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
            pad_npts = max_length - current_len
            
            if pad_mode == 'noise':
                # Generate spectrum-matched noise for padding
                reference_signal = find_reference_signal(trace.data)
                pad_data = generate_matching_noise(reference_signal, pad_npts)
                trace.data = np.concatenate([trace.data, pad_data.astype(trace.data.dtype)])
            elif pad_mode == 'zero':
                trace.data = np.concatenate([
                    trace.data,
                    np.zeros(pad_npts, dtype=trace.data.dtype)
                ])
            else:  # 'repeat_last' or default fallback
                last_val = trace.data[-1] if current_len > 0 else 0
                trace.data = np.concatenate([
                    trace.data, 
                    np.full(pad_npts, last_val, dtype=trace.data.dtype)
                ])
        else:
            trace.data = trace.data[:max_length]
    
    # Clean up any temporary arrays
    gc.collect()
    return wf

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


def generate_matching_noise(reference_signal, output_length):
    """
    Generate noise that matches the spectral characteristics of reference_signal.
    Uses spectral shaping to create realistic seismic background noise.
    
    Args:
        reference_signal: Array of reference waveform data
        output_length: Length of output noise array
        
    Returns:
        Noise array with matched spectral characteristics
    """
    if len(reference_signal) < 50:
        return np.random.normal(np.mean(reference_signal), 
                                max(np.std(reference_signal), 1e-6), output_length)
    
    ref_fft = np.fft.rfft(reference_signal)
    ref_amplitude = np.abs(ref_fft)
    white_noise = np.random.normal(0, 1, output_length)
    noise_fft = np.fft.rfft(white_noise)
    
    ref_freqs = np.linspace(0, 1, len(ref_amplitude))
    noise_freqs = np.linspace(0, 1, len(noise_fft))
    interp_amplitude = np.interp(noise_freqs, ref_freqs, ref_amplitude)
    
    shaped_fft = noise_fft * interp_amplitude / (np.abs(noise_fft) + 1e-10)
    shaped_noise = np.fft.irfft(shaped_fft, n=output_length)
    
    shaped_noise = shaped_noise - np.mean(shaped_noise)
    if np.std(shaped_noise) > 1e-10:
        shaped_noise = shaped_noise / np.std(shaped_noise) * np.std(reference_signal)
    shaped_noise = shaped_noise + np.mean(reference_signal)
    
    return shaped_noise


def find_reference_signal(wf_data, window_size=500, max_search=5000, min_unique=300):
    """
    Find a non-flat reference region in waveform data for noise generation.
    
    Args:
        wf_data: Waveform data array
        window_size: Size of window to check
        max_search: Maximum samples to search
        min_unique: Minimum unique values to consider non-flat
        
    Returns:
        Reference signal array
    """
    for start_idx in range(0, min(len(wf_data) - window_size, max_search), 100):
        sample_window = wf_data[start_idx:start_idx + window_size]
        n_unique = len(np.unique(np.round(sample_window, decimals=4)))
        if n_unique > min_unique:
            return sample_window
    
    # Fallback: use beginning of data
    return wf_data[:window_size] if len(wf_data) >= window_size else wf_data


def pad_waveform_with_noise(wf_data, pad_npts, pad_position='front'):
    """
    Pad waveform with spectrum-matched background noise.
    
    Args:
        wf_data: Original waveform data
        pad_npts: Number of samples to pad
        pad_position: 'front' or 'back'
        
    Returns:
        Padded waveform data
    """
    reference_signal = find_reference_signal(wf_data)
    pad_noise = generate_matching_noise(reference_signal, pad_npts)
    
    if pad_position == 'front':
        return np.concatenate([pad_noise, wf_data])
    else:
        return np.concatenate([wf_data, pad_noise])


def fill_flat_regions(data, reference_signal=None, window_size=100, min_unique=50):
    """
    Fill flat/constant regions in waveform data with spectrum-matched noise.
    
    Args:
        data: Waveform data array
        reference_signal: Reference signal for noise generation (optional)
        window_size: Size of window to check for flat regions
        min_unique: Minimum unique values to consider non-flat
        
    Returns:
        Data with flat regions filled with noise
    """
    if reference_signal is None:
        reference_signal = find_reference_signal(data)
    
    filled_data = data.copy()
    for j in range(0, len(filled_data) - window_size, window_size):
        segment = filled_data[j:j+window_size]
        n_unique = len(np.unique(np.round(segment, decimals=4)))
        if n_unique < min_unique:
            noise_fill = generate_matching_noise(reference_signal, window_size)
            filled_data[j:j+window_size] = noise_fill
    
    return filled_data
