#!/usr/bin/env python3
"""
Core REDPAN Implementation
=========================

This module contains the main REDPAN class that implements
SeisBench-style direct array accumulation for high-performance
continuous seismic phase picking.

FIXED: Time shift issue resolved by using spectrum-matched noise padding
instead of zero/median padding, and proper truncation of padded regions.
"""

import gc
import logging
import numpy as np
import tensorflow as tf
from copy import deepcopy
from typing import Tuple
from obspy import Stream
from redpan.utils import (
    sac_len_complement, stream_standardize,
    pad_waveform_with_noise, generate_matching_noise, find_reference_signal
)
from redpan.picks import pred_postprocess

logger = logging.getLogger(__name__)


class REDPAN:
    """
    True SeisBench-style RED-PAN implementation with direct array accumulation
    
    Uses spectrum-matched noise padding to ensure proper time alignment
    in sliding window predictions.
    """
    
    def __init__(self, 
                 model,
                 pred_npts: int = 6000,
                 dt: float = 0.01,
                 pred_interval_sec: float = 10.0,
                 batch_size: int = 32):
        """
        Initialize the RED-PAN picker
        
        Args:
            model: TensorFlow model for prediction
            pred_npts: Model input length (samples)
            dt: Sample interval (seconds)
            pred_interval_sec: Sliding window step (seconds)
            batch_size: Batch size for prediction
        """
        self.model = model
        self.pred_npts = pred_npts
        self.dt = dt
        self.pred_interval_sec = pred_interval_sec
        self.batch_size = batch_size
        
        # Calculate prediction interval in samples
        self.pred_interval_pt = int(round(pred_interval_sec / dt))
        
        # Use uniform weights for accumulation (matches legacy median filter)
        self.position_weights = np.ones(self.pred_npts, dtype=np.float32)
        
        gc.collect()
        
        logger.info(f"REDPAN initialized: pred_npts={pred_npts}, "
                   f"pred_interval_sec={pred_interval_sec}, batch_size={batch_size}")
    
    def _pad_stream_with_noise(self, wf: Stream, pad_npts: int) -> Stream:
        """
        Pad stream with spectrum-matched noise at both ends.
        
        Args:
            wf: Input ObsPy stream
            pad_npts: Number of samples to pad at each end
            
        Returns:
            Padded stream
        """
        wf_padded = wf.copy()
        
        for trace in wf_padded:
            # Find reference signal for noise generation
            ref_signal = find_reference_signal(trace.data)
            
            # Generate noise for front and back padding
            front_noise = generate_matching_noise(ref_signal, pad_npts)
            back_noise = generate_matching_noise(ref_signal, pad_npts)
            
            # Pad the trace data
            trace.data = np.concatenate([front_noise, trace.data, back_noise])
            
            # Adjust starttime to account for front padding
            trace.stats.starttime -= pad_npts * self.dt
        
        return wf_padded
    
    def _prepare_waveform_slices(self, wf: Stream) -> np.ndarray:
        """
        Prepare waveform slices for sliding window prediction.
        
        The waveform should already be padded. This method extracts
        overlapping windows with proper normalization.
        
        Args:
            wf: Padded ObsPy stream
            
        Returns:
            Array of normalized waveform slices (n_windows, pred_npts, 3)
        """
        data_len = len(wf[0].data)
        
        # Calculate number of windows
        n_windows = (data_len - self.pred_npts) // self.pred_interval_pt + 1
        
        logger.debug(f"Preparing {n_windows} windows from {data_len} samples")
        
        # Extract and normalize slices for each channel
        wf_channels = []
        for ch in range(3):
            channel_data = wf[ch].data
            slices = []
            
            for i in range(n_windows):
                start = i * self.pred_interval_pt
                end = start + self.pred_npts
                
                window = channel_data[start:end].copy()
                
                # Normalize: demean and standardize
                window = window - np.mean(window)
                std = np.std(window)
                if std > 1e-10:
                    window = window / std
                
                slices.append(window)
            
            wf_channels.append(np.array(slices))
            
            del slices
            gc.collect()
        
        # Stack channels: (n_windows, pred_npts, 3)
        wf_slices = np.stack(wf_channels, axis=-1)
        
        del wf_channels
        gc.collect()
        
        return wf_slices
    
    def _batch_predict(self, wf_slices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run batch prediction on waveform slices.
        
        Args:
            wf_slices: Array of shape (n_windows, pred_npts, 3)
            
        Returns:
            Tuple of (predictions, masks) arrays
        """
        n_slices = len(wf_slices)
        n_batches = (n_slices + self.batch_size - 1) // self.batch_size
        
        all_predictions = []
        all_masks = []
        
        logger.debug(f"Running {n_slices} slices in {n_batches} batches")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_slices)
            batch_data = wf_slices[start_idx:end_idx]
            
            # Run prediction
            pred_result = self.model.predict(batch_data, verbose=0)
            
            # Parse model output
            if isinstance(pred_result, (list, tuple)) and len(pred_result) == 2:
                predictions, masks = pred_result
            else:
                predictions = pred_result
                masks = np.ones_like(predictions[:, :, 0:1])
            
            all_predictions.append(predictions)
            all_masks.append(masks)
            
            del batch_data, pred_result
            if batch_idx % 25 == 0 and batch_idx > 0:
                gc.collect()
        
        final_predictions = np.concatenate(all_predictions, axis=0)
        final_masks = np.concatenate(all_masks, axis=0)
        
        del all_predictions, all_masks
        gc.collect()
        
        logger.debug(f"Batch prediction completed: {final_predictions.shape}")
        return final_predictions, final_masks
    
    def _accumulate_predictions(self, 
                                predictions: np.ndarray, 
                                masks: np.ndarray,
                                total_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Accumulate sliding window predictions using weighted averaging.
        
        This is a clean implementation that directly accumulates predictions
        into an output array of the correct size.
        
        Args:
            predictions: Array of shape (n_windows, pred_npts, 2) for P and S
            masks: Array of shape (n_windows, pred_npts, 1)
            total_samples: Total length of output (padded waveform length)
            
        Returns:
            Tuple of (P_pred, S_pred, M_pred) arrays
        """
        # Pre-allocate accumulation arrays
        P_acc = np.zeros(total_samples, dtype=np.float32)
        S_acc = np.zeros(total_samples, dtype=np.float32)
        M_acc = np.zeros(total_samples, dtype=np.float32)
        W_acc = np.zeros(total_samples, dtype=np.float32)
        
        n_windows = len(predictions)
        logger.debug(f"Accumulating {n_windows} windows into {total_samples} samples")
        
        for i in range(n_windows):
            start_pos = i * self.pred_interval_pt
            end_pos = start_pos + self.pred_npts
            
            # Boundary check
            if end_pos > total_samples:
                end_pos = total_samples
                actual_len = end_pos - start_pos
                if actual_len <= 0:
                    continue
                weights = self.position_weights[:actual_len]
                pp = predictions[i, :actual_len, 0]
                ss = predictions[i, :actual_len, 1]
                mm = masks[i, :actual_len, 0]
            else:
                weights = self.position_weights
                pp = predictions[i, :, 0]
                ss = predictions[i, :, 1]
                mm = masks[i, :, 0]
            
            # Accumulate with weights
            P_acc[start_pos:end_pos] += pp * weights
            S_acc[start_pos:end_pos] += ss * weights
            M_acc[start_pos:end_pos] += mm * weights
            W_acc[start_pos:end_pos] += weights
        
        # Normalize by weights
        W_acc = np.maximum(W_acc, 1e-8)
        P_pred = P_acc / W_acc
        S_pred = S_acc / W_acc
        M_pred = M_acc / W_acc
        
        del P_acc, S_acc, M_acc, W_acc
        gc.collect()
        
        return P_pred, S_pred, M_pred
    
    def predict(self, wf: Stream, postprocess: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main prediction function with proper time alignment.
        
        Strategy:
        - If waveform <= model receptive field: Direct single-window prediction
        - If waveform > model receptive field: 
          1. Pad waveform with spectrum-matched noise (pred_npts at each end)
          2. Run sliding window accumulation
          3. Truncate padding from predictions
        
        Args:
            wf: Input waveform stream (3-component)
            postprocess: Whether to apply postprocessing
            
        Returns:
            Tuple of (P_predictions, S_predictions, Mask_predictions)
        """
        original_npts = len(wf[0].data)
        
        # Case 1: Short waveform - direct prediction (no sliding window)
        # Use direct prediction if waveform fits within model receptive field (+ 1 second tolerance)
        if original_npts <= self.pred_npts + 100:
            logger.debug(f"Short waveform ({original_npts} samples): using direct prediction")
            
            # Simply pad with zeros at the end to reach model input size
            _wf = wf.copy()
            for tr in _wf:
                data = tr.data.astype(np.float32)
                if len(data) < self.pred_npts:
                    # Pad with zeros at the end
                    data = np.pad(data, (0, self.pred_npts - len(data)), mode='constant', constant_values=0)
                else:
                    # Truncate if slightly longer
                    data = data[:self.pred_npts]
                tr.data = data
            
            _wf = stream_standardize(_wf, data_length=self.pred_npts)
            
            # Single prediction
            batch_data = np.stack([W.data for W in _wf], axis=-1)[np.newaxis, ...]
            picks, masks = self.model.predict(batch_data, verbose=0)
            
            # Extract predictions - truncate or pad to match original length
            if original_npts <= self.pred_npts:
                array_P = picks[0, :original_npts, 0]
                array_S = picks[0, :original_npts, 1]
                array_M = masks[0, :original_npts, 0]
            else:
                # Pad with zeros for samples beyond pred_npts
                extra_samples = original_npts - self.pred_npts
                array_P = np.concatenate([picks[0, :, 0], np.zeros(extra_samples, dtype=np.float32)])
                array_S = np.concatenate([picks[0, :, 1], np.zeros(extra_samples, dtype=np.float32)])
                array_M = np.concatenate([masks[0, :, 0], np.zeros(extra_samples, dtype=np.float32)])
            
            del batch_data, picks, masks, _wf
            gc.collect()
        
        # Case 2: Long waveform - sliding window with noise padding
        else:
            logger.debug(f"Long waveform ({original_npts} samples): using sliding window")
            
            # Pad with spectrum-matched noise at both ends
            pad_npts = self.pred_npts
            wf_padded = self._pad_stream_with_noise(wf, pad_npts)
            padded_len = len(wf_padded[0].data)
            
            logger.debug(f"Padded: {original_npts} -> {padded_len} samples (pad={pad_npts})")
            
            # Prepare slices and run prediction
            wf_slices = self._prepare_waveform_slices(wf_padded)
            predictions, masks = self._batch_predict(wf_slices)
            
            del wf_slices, wf_padded
            gc.collect()
            
            # Accumulate predictions
            P_padded, S_padded, M_padded = self._accumulate_predictions(
                predictions, masks, padded_len
            )
            
            del predictions, masks
            gc.collect()
            
            # Truncate padding: keep only [pad_npts : pad_npts + original_npts]
            array_P = P_padded[pad_npts:pad_npts + original_npts]
            array_S = S_padded[pad_npts:pad_npts + original_npts]
            array_M = M_padded[pad_npts:pad_npts + original_npts]
            
            del P_padded, S_padded, M_padded
            gc.collect()
        
        # Handle NaN/Inf values
        invalid_mask = np.isnan(array_M) | np.isinf(array_M)
        array_P[invalid_mask] = 0.0
        array_S[invalid_mask] = 0.0
        array_M[invalid_mask] = 0.0
        
        # Apply postprocessing if requested
        if postprocess and hasattr(self, 'postprocess_config') and self.postprocess_config:
            array_P, array_S, array_M = pred_postprocess(
                array_P, array_S, array_M,
                dt=self.dt,
                **self.postprocess_config,
            )
        
        return array_P, array_S, array_M
    
    def annotate_stream(self, wf: Stream, postprocess: bool = False) -> Tuple[Stream, Stream, Stream]:
        """
        Annotate stream with REDPAN predictions, creating ObsPy streams.
        
        Args:
            wf: Input ObsPy stream (3-component seismic data)
            postprocess: Whether to apply postprocessing
            
        Returns:
            Tuple of (P_stream, S_stream, M_stream) as ObsPy Stream objects
        """
        # Get predictions
        array_P, array_S, array_M = self.predict(wf, postprocess=postprocess)
        
        # Create output streams
        P_stream, S_stream, M_stream = Stream(), Stream(), Stream()
        
        W_data = [array_P, array_S, array_M]
        W_chn = ["redpan_P", "redpan_S", "redpan_mask"]
        W_sac = [P_stream, S_stream, M_stream]
        
        for k in range(3):
            W = deepcopy(wf[0])
            W.data = W_data[k]
            W.stats.channel = W_chn[k]
            W_sac[k].append(W)
        
        # Slice to match original time window (safety check)
        P_stream = P_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        S_stream = S_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        M_stream = M_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        
        del wf, array_P, array_S, array_M, W_data, W_sac
        gc.collect()
        
        logger.debug(f"Created annotated streams: P={len(P_stream[0].data)}, "
                    f"S={len(S_stream[0].data)}, M={len(M_stream[0].data)} samples")
        
        return P_stream, S_stream, M_stream


# Legacy compatibility classes and functions below
# =================================================

class PhasePicker:
    """
    Legacy PhasePicker class for backward compatibility.
    """
    
    def __init__(
        self,
        model=None,
        dt=0.01,
        pred_npts=3000,
        pred_interval_sec=10,
        STMF_max_sec=1200,
        postprocess_config={
            "mask_trigger": [0.1, 0.1],
            "mask_len_thre": 0.5,
            "mask_err_win": 0.5,
            "detection_threshold": 0.3,
            "P_threshold": 0.1,
            "S_threshold": 0.1
        },
    ):
        self.model = model
        self.dt = dt
        self.pred_npts = pred_npts
        self.pred_interval_sec = pred_interval_sec
        self.STMF_max_sec = STMF_max_sec
        self.postprocess_config = postprocess_config

        if model is None:
            raise AssertionError("The Phase picker model should be defined!")
        
        # Use new REDPAN implementation internally
        self._picker = REDPAN(
            model=model,
            pred_npts=pred_npts,
            dt=dt,
            pred_interval_sec=pred_interval_sec,
            batch_size=32
        )
        self._picker.postprocess_config = postprocess_config

    def predict(self, wf=None, postprocess=False):
        if wf is None:
            raise AssertionError("Obspy.stream should be assigned as `wf=?`!")
        return self._picker.predict(wf, postprocess=postprocess)
    
    def annotate_stream(self, wf, STMF_max_sec=None, postprocess=False):
        return self._picker.annotate_stream(wf, postprocess=postprocess)


def conti_standard_wf_fast(wf, pred_npts, pred_interval_sec, dt, pad_zeros=True):
    """
    Legacy waveform preparation function - kept for compatibility.
    """
    from redpan.utils import sac_len_complement
    from copy import deepcopy
    
    raw_n = len(wf[0].data)
    pred_rate = int(pred_interval_sec / dt)
    full_len = int(pred_npts + pred_rate * np.ceil(raw_n - pred_npts) / pred_rate)
    n_marching_win = int((full_len - pred_npts) / pred_rate) + 1

    wf = sac_len_complement(wf.copy(), max_length=full_len)
    pad_bef = pred_npts - pred_rate
    pad_aft = pred_npts
    for W in wf:
        W.data = np.insert(W.data, 0, np.zeros(pad_bef))
        W.data = np.insert(W.data, len(W.data), np.zeros(pad_aft))

    wf_n = []
    for w in range(3):
        wf_ = np.array([
            deepcopy(wf[w].data[pred_rate * i : pred_rate * i + pred_npts])
            for i in range(n_marching_win)
        ])
        wf_dm = np.array([i - np.mean(i) for i in wf_])
        wf_std = np.array([np.std(i) for i in wf_dm])
        wf_std[wf_std == 0] = 1
        wf_norm = np.array([wf_dm[i] / wf_std[i] for i in range(len(wf_dm))])
        wf_n.append(wf_norm)

    wf_slices = np.stack([wf_n[0], wf_n[1], wf_n[2]], -1)
    return np.array(wf_slices), pad_bef, pad_aft


def pred_MedianFilter(preds, masks, wf_npts, dt, pred_npts, pred_interval_sec, pad_bef, pad_aft):
    """
    Legacy median filter function - kept for compatibility.
    """
    wf_n = wf_npts + (pad_bef + pad_aft)
    pred_array_P = [[] for _ in range(wf_n)]
    pred_array_S = [[] for _ in range(wf_n)]
    pred_array_mask = [[] for _ in range(wf_n)]
    pred_interval_pt = int(round(pred_interval_sec / dt))

    init_pt = 0
    for i in range(len(preds)):
        pp = np.array_split(preds[i].T[0], pred_npts)
        ss = np.array_split(preds[i].T[1], pred_npts)
        mm = np.array_split(masks[i].T[0], pred_npts)
        j = 0
        for p, s, m in zip(pp, ss, mm):
            pred_array_P[init_pt + j].append(p)
            pred_array_S[init_pt + j].append(s)
            pred_array_mask[init_pt + j].append(m)
            j += 1
        init_pt += pred_interval_pt

    pred_array_P = np.array(pred_array_P, dtype="object")
    pred_array_S = np.array(pred_array_S, dtype="object")
    pred_array_mask = np.array(pred_array_mask, dtype="object")
    
    lenP = np.array([len(p) for p in pred_array_P])
    nums = np.unique(lenP)
    array_P_med = np.zeros(wf_n)
    array_S_med = np.zeros(wf_n)
    array_M_med = np.zeros(wf_n)
    
    for k in nums:
        num_idx = np.where(lenP == k)[0]
        array_P_med[num_idx] = np.median(np.hstack(np.take(pred_array_P, num_idx)), axis=0)
        array_S_med[num_idx] = np.median(np.hstack(np.take(pred_array_S, num_idx)), axis=0)
        array_M_med[num_idx] = np.median(np.hstack(np.take(pred_array_mask, num_idx)), axis=0)
    
    del pred_array_P, pred_array_S, pred_array_mask
    gc.collect()

    array_P_med = array_P_med[pad_bef:-pad_aft]
    array_S_med = array_S_med[pad_bef:-pad_aft]
    array_M_med = array_M_med[pad_bef:-pad_aft]
    assert len(array_P_med) == wf_npts

    return array_P_med, array_S_med, array_M_med
