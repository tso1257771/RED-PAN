#!/usr/bin/env python3
"""
Core REDPAN Implementation
=========================

This module contains the main REDPAN class that implements
SeisBench-style direct array accumulation for high-performance
continuous seismic phase picking.
"""

import gc
import logging
import numpy as np
import tensorflow as tf
from copy import deepcopy
from typing import Tuple
from obspy import Stream
from redpan.utils import sac_len_complement, stream_standardize
from redpan.picks import pred_postprocess

logger = logging.getLogger(__name__)


class REDPAN:
    """
    True SeisBench-style RED-PAN implementation with direct array accumulation
    
    This eliminates the list-based accumulation that makes the original slow.
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
            pred_npts: Model input length
            dt: Sample rate
            pred_interval_sec: Sliding window step
            batch_size: Batch size for prediction
        """
        self.model = model
        self.pred_npts = pred_npts
        self.dt = dt
        self.pred_interval_sec = pred_interval_sec
        self.batch_size = batch_size
        
        # Calculate prediction interval in samples
        self.pred_interval_pt = int(round(pred_interval_sec / dt))
        
        # Pre-compute Gaussian position weights (SeisBench style)
        self.position_weights = self._create_gaussian_weights()
        
        # Force garbage collection after initialization
        gc.collect()
        
        logger.info(f"REDPAN initialized: "
                   f"pred_npts={pred_npts}, batch_size={batch_size}")
        logger.debug(f"Using Gaussian position weights with shape {self.position_weights.shape}")
    
    def _create_gaussian_weights(self) -> np.ndarray:
        """
        Create Gaussian position weights exactly like SeisBench
        
        Returns:
            Gaussian weights centered on prediction window
        """
        center = self.pred_npts // 2
        sigma = self.pred_npts / 6.0  # SeisBench default
        positions = np.arange(self.pred_npts)
        weights = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
        
        # Normalize to prevent weight inflation
        weights = weights / np.mean(weights)
        
        logger.debug(f"Created Gaussian weights: center={center}, sigma={sigma:.1f}")
        return weights.astype(np.float32)
    
    
    def _prepare_waveform_slices(self, wf: Stream) -> Tuple[np.ndarray, int, int]:
        """
        Prepare waveform slices for RED-PAN
        """

        raw_n = len(wf[0].data)
        pred_rate = int(self.pred_interval_sec / self.dt)
        
        full_len = int(self.pred_npts + pred_rate * np.ceil(raw_n - self.pred_npts) / pred_rate)
        n_marching_win = int((full_len - self.pred_npts) / pred_rate) + 1
        
        # Complement waveform length
        wf_copy = sac_len_complement(wf.copy(), max_length=full_len)
        
        # Handle padding
        pad_value = np.median(wf_copy[0].data)
        pad_bef = self.pred_npts - pred_rate
        pad_aft = self.pred_npts
        
        for W in wf_copy:
            W.data = np.insert(W.data, 0, np.full(pad_bef, pad_value))
            W.data = np.insert(W.data, len(W.data), np.full(pad_aft, pad_value))
        
        # Generate slices with normalization
        wf_n = []
        for w in range(3):
            wf_ = np.array([
                deepcopy(wf_copy[w].data[pred_rate * i : pred_rate * i + self.pred_npts])
                for i in range(n_marching_win)
            ])
            # Apply normalization per slice
            wf_dm = np.array([i - np.mean(i) for i in wf_])
            wf_std = np.array([np.std(i) for i in wf_dm])
            # Prevent ZeroDivisionError
            wf_std[wf_std == 0] = 1  
            wf_norm = np.array([wf_dm[i] / wf_std[i] for i in range(len(wf_dm))])
            wf_n.append(wf_norm)
            
            # Clean up intermediate arrays for this channel
            del wf_, wf_dm, wf_std, wf_norm
            gc.collect()

        # Stack waveforms
        wf_slices = np.stack([wf_n[0], wf_n[1], wf_n[2]], -1)
        
        # Clean up temporary data
        del wf_n, wf_copy
        gc.collect()
        
        logger.debug(f"Prepared {len(wf_slices)} slices, padding: {pad_bef}, {pad_aft}")
        return np.array(wf_slices), pad_bef, pad_aft
    
    def _batch_predict(self, wf_slices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run batch prediction efficiently
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
            
            # Only cleanup variables
            del batch_data, pred_result
            # Less frequent cleanup - only every 25 batches
            if batch_idx % 25 == 0 and batch_idx > 0:
                gc.collect()
        
        # Concatenate results
        final_predictions = np.concatenate(all_predictions, axis=0)
        final_masks = np.concatenate(all_masks, axis=0)
        
        # FIXED: Only cleanup variables, keep TensorFlow session alive
        del all_predictions, all_masks
        
        logger.debug(f"Batch prediction completed: {final_predictions.shape}")
        return final_predictions, final_masks
    
    def _seisbench_accumulation(self, 
                               predictions: np.ndarray, 
                               masks: np.ndarray,
                               wf_npts: int,
                               pad_before: int,
                               pad_after: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SeisBench-style direct array accumulation
        
        This is the key optimization: NO list-based accumulation!
        Direct array operations only.
        """
        # Calculate full padded length
        wf_n = wf_npts + pad_before + pad_after
        
        # Pre-allocate accumulation arrays (FLOAT32 for speed)
        P_accumulator = np.zeros(wf_n, dtype=np.float32)
        S_accumulator = np.zeros(wf_n, dtype=np.float32)
        M_accumulator = np.zeros(wf_n, dtype=np.float32)
        weight_accumulator = np.zeros(wf_n, dtype=np.float32)
        
        logger.debug(f"SeisBench accumulation: {len(predictions)} predictions → {wf_n} samples")
        
        # Process each prediction slice with DIRECT accumulation
        for i in range(len(predictions)):
            # Calculate position for this prediction
            start_pos = i * self.pred_interval_pt
            end_pos = start_pos + self.pred_npts
            
            # Handle boundary conditions
            if end_pos > wf_n:
                end_pos = wf_n
                actual_len = end_pos - start_pos
                if actual_len <= 0:
                    continue
                pred = predictions[i][:actual_len]
                mask = masks[i][:actual_len]
                weights = self.position_weights[:actual_len]
            else:
                pred = predictions[i]
                mask = masks[i]
                weights = self.position_weights
            
            # Extract P and S predictions
            if pred.shape[-1] >= 2:
                pp = pred[:, 0]  # P predictions
                ss = pred[:, 1]  # S predictions
            else:
                pp = pred.flatten()
                ss = np.zeros_like(pp)
            
            # Extract mask
            if len(mask.shape) >= 2:
                mm = mask[:, 0]
            else:
                mm = mask.flatten()
            
            # DIRECT ACCUMULATION (This is the key SeisBench optimization!)
            # No lists, no appends, just direct array operations
            P_accumulator[start_pos:end_pos] += pp * weights
            S_accumulator[start_pos:end_pos] += ss * weights
            M_accumulator[start_pos:end_pos] += mm * weights
            weight_accumulator[start_pos:end_pos] += weights
            
            # Clean up temporary variables every iteration to prevent memory buildup
            if i % 100 == 0:
                gc.collect()
        
        # Vectorized normalization (avoid division by zero)
        weight_accumulator = np.maximum(weight_accumulator, 1e-8)
        
        P_final = P_accumulator / weight_accumulator
        S_final = S_accumulator / weight_accumulator
        M_final = M_accumulator / weight_accumulator
        

        
        # Remove padding exactly like original
        P_result = P_final[pad_before:-pad_after] if pad_after > 0 else P_final[pad_before:]
        S_result = S_final[pad_before:-pad_after] if pad_after > 0 else S_final[pad_before:]
        M_result = M_final[pad_before:-pad_after] if pad_after > 0 else M_final[pad_before:]
        
        # Clean up final intermediate arrays
        del P_accumulator, S_accumulator, M_accumulator, weight_accumulator, P_final, S_final, M_final
        gc.collect()
        
        # Validate output length
        assert len(P_result) == wf_npts, f"Length mismatch: {len(P_result)} != {wf_npts}"
        
        logger.debug(f"SeisBench accumulation completed: output length {len(P_result)}")
        return P_result, S_result, M_result
    
    def predict(self, wf: Stream, postprocess: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main prediction function - true SeisBench-style processing
        
        Args:
            wf: Input waveform stream
            postprocess: Whether to apply postprocessing
            
        Returns:
            Tuple of (P_predictions, S_predictions, Mask_predictions)
        """
        if len(wf[0].data) <= self.pred_npts+1:
            logging.warning(f"Data length <= than {self.pred_npts} points.")
            wf_npts = len(wf[0].data)
            # This will simple fill value after the end of the waveform
            _wf = sac_len_complement(wf.copy(), max_length=self.pred_npts)
            _wf = stream_standardize(_wf, data_length=self.pred_npts)
            
            batch_data = np.stack(
                [W.data for W in _wf], -1)[np.newaxis, ...]
            picks, masks = self.model.predict(batch_data, verbose=0)
            array_P_med = picks[0].T[0][:wf_npts]
            array_S_med = picks[0].T[1][:wf_npts]
            array_M_med = masks[0].T[0][:wf_npts]
            # Clean up predictions
            del batch_data, picks, masks
        else:
            # Step 1: Prepare waveform slices (exact original logic)
            wf_slices, pad_bef, pad_aft = self._prepare_waveform_slices(wf)
            
            # Step 2: Run batch prediction
            predictions, masks = self._batch_predict(wf_slices)
            
            # Clean up slices immediately
            del wf_slices
            
            # Step 3: TRUE SeisBench-style direct accumulation
            wf_npts = len(wf[0].data)
            array_P_med, array_S_med, array_M_med = self._seisbench_accumulation(
                predictions=predictions,
                masks=masks,
                wf_npts=wf_npts,
                pad_before=pad_bef,
                pad_after=pad_aft
            )
            
            # Clean up predictions
            del predictions, masks
        gc.collect()
        
        # Handle NaN/Inf values (same as original)
        nan_mask = np.isnan(array_M_med)
        inf_mask = np.isinf(array_M_med)
        invalid_mask = nan_mask | inf_mask
        
        array_P_med[invalid_mask] = 0.0
        array_S_med[invalid_mask] = 0.0
        array_M_med[invalid_mask] = 0.0
        
        # Clean up intermediate masks
        del nan_mask, inf_mask, invalid_mask
        
        # Apply postprocessing if requested
        if postprocess and hasattr(self, 'postprocess_config') and self.postprocess_config:
            array_P_med, array_S_med, array_M_med = pred_postprocess(
                array_P_med,
                array_S_med,
                array_M_med,
                dt=self.dt,
                **self.postprocess_config,
            )
        
        # Final cleanup before returning results
        return array_P_med, array_S_med, array_M_med

    def _output_stream_postprocess(self, P_stream, S_stream, M_stream, postprocess_config=None):
        """
        Apply postprocessing to the output arrays.
        """
        P_stream_post = deepcopy(P_stream)
        S_stream_post = deepcopy(S_stream)
        M_stream_post = deepcopy(M_stream)
        array_P = P_stream_post[0].data
        array_S = S_stream_post[0].data
        array_M = M_stream_post[0].data
        array_P, array_S, array_M = pred_postprocess(
            array_P,
            array_S,
            array_M,
            dt=self.dt,
            **self.postprocess_config,
        )
        P_stream_post[0].data = array_P
        S_stream_post[0].data = array_S
        M_stream_post[0].data = array_M
        return P_stream_post, S_stream_post, M_stream_post

    def annotate_stream(self, wf: Stream, postprocess: bool = False) -> Tuple[Stream, Stream, Stream]:
        """
        Annotate stream with REDPAN predictions, creating ObsPy streams
        
        This function creates ObsPy streams for P, S, and mask predictions
        analogous to the PhasePicker's annotate_stream method.
        
        Args:
            wf: Input ObsPy stream (3-component seismic data)
            postprocess: Whether to apply postprocessing
            
        Returns:
            Tuple of (P_stream, S_stream, M_stream) as ObsPy Stream objects
        """
        from copy import deepcopy
        
        # Get predictions using the main predict function
        array_P, array_S, array_M = self.predict(
            wf, postprocess=postprocess)
        
        # Create empty streams for P, S, and mask predictions
        P_stream, S_stream, M_stream = Stream(), Stream(), Stream()
        
        # Create traces for each prediction type
        W_data = [array_P, array_S, array_M]
        W_chn = ["redpan_P", "redpan_S", "redpan_mask"]
        W_sac = [P_stream, S_stream, M_stream]
        
        # Create traces based on the first trace of input waveform
        for k in range(3):
            W = deepcopy(wf[0])  # Use first trace as template
            W.data = W_data[k]   # Replace data with predictions
            W.stats.channel = W_chn[k]  # Set appropriate channel name
            W_sac[k].append(W)   # Add to corresponding stream
        
        # Ensure streams match the original time window
        P_stream = P_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        S_stream = S_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        M_stream = M_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        
        # Clean up temporary arrays
        del wf, array_P, array_S, array_M, W_data, W_sac

        logger.debug(f"Created annotated streams: P={len(P_stream[0].data)}, "
                    f"S={len(S_stream[0].data)}, M={len(M_stream[0].data)} samples")
        
        return P_stream, S_stream, M_stream


class PhasePicker:
    """
    Legacy PhasePicker class for backward compatibility.
    This class provides the same interface as the original RED-PAN
    but uses the improved REDPAN implementation internally.
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
        self.STMF_max_sec = STMF_max_sec # Seismogram-Tracking Median Filter length
        self.postprocess_config = postprocess_config

        if model == None:
            AssertionError("The Phase picker model should be defined!")

    def predict(self, wf=None, postprocess=False):
        from time import time

        if wf == None:
            AssertionError("Obspy.stream should be assigned as `wf=?`!")
        if len(wf[0].data) < self.pred_npts:
            AssertionError(f"Data should be longer than {self.pred_npts} points.")
        ## store continuous data into array according to prediction interval
        wf_slices, pad_bef, pad_aft = conti_standard_wf_fast(
            wf,
            pred_npts=self.pred_npts,
            pred_interval_sec=self.pred_interval_sec,
            dt=self.dt,
        )
        ## make prediction
        t1 = time()
        predPhase, masks = self.model.predict(wf_slices)
        del wf_slices
        gc.collect()
        # print(f"Prediction making: {time()-t1:.2f} secs")
        ## apply median filter to sliding predictions
        wf_npts = len(wf[0].data)
        array_P_med, array_S_med, array_M_med = pred_MedianFilter(
            preds=predPhase,
            masks=masks,
            wf_npts=wf_npts,
            dt=self.dt,
            pred_npts=self.pred_npts,
            pred_interval_sec=self.pred_interval_sec,
            pad_bef=pad_bef,
            pad_aft=pad_aft,
        )

        del predPhase
        del masks
        gc.collect()
        # replace nan by 0
        find_nan = np.where(np.isnan(array_M_med))[0]
        array_P_med[find_nan] = np.zeros(len(find_nan))
        array_S_med[find_nan] = np.zeros(len(find_nan))
        array_M_med[find_nan] = np.zeros(len(find_nan))
        # replace inf by 0
        find_inf = np.where(np.isnan(array_M_med))[0]
        array_P_med[find_inf] = np.zeros(len(find_inf))
        array_S_med[find_inf] = np.zeros(len(find_inf))
        array_M_med[find_inf] = np.zeros(len(find_inf))

        if postprocess:
            from redpan.picks import pred_postprocess
            array_P_med, array_S_med, array_M_med = pred_postprocess(
                array_P_med,
                array_S_med,
                array_M_med,
                dt=self.dt,
                **self.postprocess_config,
            )

        return array_P_med, array_S_med, array_M_med
    
    def annotate_stream(self, wf, STMF_max_sec=None, postprocess=False):
        wf_stt = wf[0].stats.starttime
        if not STMF_max_sec:
            STMF_max_sec = self.STMF_max_sec
        # < Case 1 >
        # if data samples are smaller than model receptive field (npts),
        # append the data to the same length and make predictions
        if wf[0].stats.npts <= self.pred_npts:
            _wf = deepcopy(wf)
            _wf = stream_standardize(_wf, data_length=self.pred_npts)
            P_stream, S_stream, M_stream = Stream(), Stream(), Stream()
            array_pick, array_mask = self.model(
                np.stack([W.data for W in _wf], -1)[np.newaxis, ...]
            )
            array_p, array_s = array_pick[0].numpy().T[:2]
            array_m = array_mask[0].numpy().T[0]

            gc.collect()
            # replace nan by 0
            find_nan = np.where(np.isnan(array_m))[0]
            array_p[find_nan] = np.zeros(len(find_nan))
            array_s[find_nan] = np.zeros(len(find_nan))
            array_m[find_nan] = np.zeros(len(find_nan))
            # replace inf by 0
            find_inf = np.where(np.isnan(array_m))[0]
            array_p[find_inf] = np.zeros(len(find_inf))
            array_s[find_inf] = np.zeros(len(find_inf))
            array_m[find_inf] = np.zeros(len(find_inf))

            if postprocess:
                from redpan.picks import pred_postprocess
                array_p, array_s, array_m = pred_postprocess(
                    array_p,
                    array_s,
                    array_m,
                    dt=self.dt,
                    **self.postprocess_config,
                )

            W_data = [array_p, array_s, array_m]
            W_chn = ["redpan_P", "redpan_S", "redpan_mask"]
            W_sac = [P_stream, S_stream, M_stream]
            for k in range(3):
                W = deepcopy(_wf[0])
                W.data = W_data[k]
                W.stats.channel = W_chn[k]
                W_sac[k].append(W)
            P_stream = P_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
            S_stream = S_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
            M_stream = M_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
            return P_stream, S_stream, M_stream

        # < Case 2 >
        # Data samples are larger than model receptive field but smaller than
        # STMF_max_sec/delta
        elif (wf[0].stats.npts > self.pred_npts) and \
            (wf[0].stats.npts < int(STMF_max_sec/wf[0].stats.delta)):

            _wf = sac_len_complement(deepcopy(wf), len(wf[0].data)+self.pred_npts)
            P_stream, S_stream, M_stream = Stream(), Stream(), Stream()

            wf_slices, pad_bef, pad_aft = conti_standard_wf_fast(
                _wf,
                pred_npts=self.pred_npts,
                pred_interval_sec=self.pred_interval_sec,
                dt=self.dt,
            )
            
            predPhase, masks = self.model.predict(wf_slices)
            del wf_slices
            gc.collect()

            # print(f"Prediction making: {time()-t1:.2f} secs")
            ## apply median filter to sliding predictions
            wf_npts = len(_wf[0].data)
            array_P_med, array_S_med, array_M_med = pred_MedianFilter(
                preds=predPhase,
                masks=masks,
                wf_npts=wf_npts,
                dt=self.dt,
                pred_npts=self.pred_npts,
                pred_interval_sec=self.pred_interval_sec,
                pad_bef=pad_bef,
                pad_aft=pad_aft,
            )

            del predPhase
            del masks
            gc.collect()
            # replace nan by 0
            find_nan = np.where(np.isnan(array_M_med))[0]
            array_P_med[find_nan] = np.zeros(len(find_nan))
            array_S_med[find_nan] = np.zeros(len(find_nan))
            array_M_med[find_nan] = np.zeros(len(find_nan))
            # replace inf by 0
            find_inf = np.where(np.isnan(array_M_med))[0]
            array_P_med[find_inf] = np.zeros(len(find_inf))
            array_S_med[find_inf] = np.zeros(len(find_inf))
            array_M_med[find_inf] = np.zeros(len(find_inf))

            if postprocess:
                from redpan.picks import pred_postprocess
                array_P_med, array_S_med, array_M_med = pred_postprocess(
                    array_P_med,
                    array_S_med,
                    array_M_med,
                    dt=self.dt,
                    **self.postprocess_config,
                )

            W_data = [array_P_med, array_S_med, array_M_med]
            W_chn = ["redpan_P", "redpan_S", "redpan_mask"]
            W_sac = [P_stream, S_stream, M_stream]
            for k in range(3):
                W = _wf[0].copy()
                W.data = W_data[k]
                W.stats.channel = W_chn[k]
                W_sac[k].append(W)
            P_stream = P_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
            S_stream = S_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
            M_stream = M_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
            return P_stream, S_stream, M_stream

        # <Case 3>
        # Data samples are larger than STMF_max_sec/delta
        seg_n = np.round(wf[0].stats.npts / int(STMF_max_sec/wf[0].stats.delta))\
            .astype(int)

        seg_wf_stt = np.array([wf_stt + STMF_max_sec*S for S in range(seg_n)])
        P_stream, S_stream, M_stream = Stream(), Stream(), Stream()
        for S in range(seg_n):
            _P_stream, _S_stream, _M_stream = Stream(), Stream(), Stream()
            if S == 0:
                seg_slice_stt = seg_wf_stt[S]
            else:
                seg_slice_stt = seg_wf_stt[S] - self.pred_npts*self.dt
            if S != seg_n - 1:
                seg_slice_ent = seg_wf_stt[S] + STMF_max_sec + self.pred_npts*self.dt
            else:
                seg_slice_ent = wf[0].stats.endtime

            _wf = sac_len_complement(
                deepcopy(wf).slice(seg_slice_stt, seg_slice_ent+self.pred_npts*self.dt),
                len(wf[0].data)+self.pred_npts
            )
            wf_slices, pad_bef, pad_aft = conti_standard_wf_fast(
                _wf,
                pred_npts=self.pred_npts,
                pred_interval_sec=self.pred_interval_sec,
                dt=self.dt,
            )
            
            predPhase, masks = self.model.predict(wf_slices)
            del wf_slices
            gc.collect()

            wf_npts = len(_wf[0].data)
            array_P_med, array_S_med, array_M_med = pred_MedianFilter(
                preds=predPhase,
                masks=masks,
                wf_npts=wf_npts,
                dt=self.dt,
                pred_npts=self.pred_npts,
                pred_interval_sec=self.pred_interval_sec,
                pad_bef=pad_bef,
                pad_aft=pad_aft,
            )

            del predPhase
            del masks
            gc.collect()
            # replace nan by 0
            find_nan = np.where(np.isnan(array_M_med))[0]
            array_P_med[find_nan] = np.zeros(len(find_nan))
            array_S_med[find_nan] = np.zeros(len(find_nan))
            array_M_med[find_nan] = np.zeros(len(find_nan))
            # replace inf by 0
            find_inf = np.where(np.isnan(array_M_med))[0]
            array_P_med[find_inf] = np.zeros(len(find_inf))
            array_S_med[find_inf] = np.zeros(len(find_inf))
            array_M_med[find_inf] = np.zeros(len(find_inf))

            if postprocess:
                from redpan.picks import pred_postprocess
                array_P_med, array_S_med, array_M_med = pred_postprocess(
                    array_P_med,
                    array_S_med,
                    array_M_med,
                    dt=self.dt,
                    **self.postprocess_config,
                )

            W_data = [array_P_med, array_S_med, array_M_med]
            W_chn = ["redpan_P", "redpan_S", "redpan_mask"]
            W_sac = [_P_stream, _S_stream, _M_stream]
            for k in range(3):
                W = _wf[0].copy()
                W.data = W_data[k]
                W.stats.channel = W_chn[k]
                W_sac[k].append(W)
            _P_stream = _P_stream.slice(_wf[0].stats.starttime, _wf[0].stats.endtime)
            _S_stream = _S_stream.slice(_wf[0].stats.starttime, _wf[0].stats.endtime)
            _M_stream = _M_stream.slice(_wf[0].stats.starttime, _wf[0].stats.endtime)
            P_stream.append(_P_stream[0])
            S_stream.append(_S_stream[0])
            M_stream.append(_M_stream[0])
        P_stream = P_stream.merge(method=1)
        S_stream = S_stream.merge(method=1)
        M_stream = M_stream.merge(method=1)

        P_stream = P_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        S_stream = S_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        M_stream = M_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)

        return P_stream, S_stream, M_stream


def conti_standard_wf_fast(wf, pred_npts, pred_interval_sec, dt, pad_zeros=True):
    """
    Core waveform preparation function for continuous processing
    
    input: 
    wf: obspy.stream object (raw_data)
    pred_npts
    pred_interval_sec
    pad_zeros: pad zeros before after the waveform for full repeating predictions 

    output:
    wf_slices (processed slices)
    pad_bef (padding before)
    pad_aft (padding after)
    """
    from redpan.utils import sac_len_complement
    from copy import deepcopy
    
    raw_n = len(wf[0].data)
    pred_rate = int(pred_interval_sec / dt)
    full_len = int(pred_npts + pred_rate * np.ceil(raw_n - pred_npts) / pred_rate)
    n_marching_win = int((full_len - pred_npts) / pred_rate) + 1
    n_padded = full_len - raw_n

    wf = sac_len_complement(wf.copy(), max_length=full_len)
    pad_bef = pred_npts - pred_rate
    pad_aft = pred_npts
    for W in wf:
        W.data = np.insert(W.data, 0, np.zeros(pad_bef))
        W.data = np.insert(W.data, len(W.data), np.zeros(pad_aft))

    wf_n = []
    for w in range(3):
        wf_ = np.array(
            [
                deepcopy(wf[w].data[pred_rate * i : pred_rate * i + pred_npts])
                for i in range(n_marching_win)
            ]
        )
        wf_dm = np.array([i - np.mean(i) for i in wf_])
        wf_std = np.array([np.std(i) for i in wf_dm])
        # reset std of 0 to 1 to prevent from ZeroDivisionError
        wf_std[wf_std == 0] = 1
        wf_norm = np.array([wf_dm[i] / wf_std[i] for i in range(len(wf_dm))])
        wf_n.append(wf_norm)

    wf_slices = np.stack([wf_n[0], wf_n[1], wf_n[2]], -1)
    return np.array(wf_slices), pad_bef, pad_aft


def pred_MedianFilter(
    preds, masks, wf_npts, dt, pred_npts, pred_interval_sec, pad_bef, pad_aft
):
    """
    Legacy median filter function for continuous predictions integration
    
    Note: This is kept for compatibility but is much slower than
    the direct array accumulation used in REDPAN class
    """
    ### 3. Integrate continuous predictions
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
    # adapt to multiprocessing

    pred_array_P = np.array(pred_array_P, dtype="object")
    pred_array_S = np.array(pred_array_S, dtype="object")
    pred_array_mask = np.array(pred_array_mask, dtype="object")
    # fast revision of bottleneck
    lenP = np.array([len(p) for p in pred_array_P])
    nums = np.unique(lenP)
    array_P_med = np.zeros(wf_n)
    array_S_med = np.zeros(wf_n)
    array_M_med = np.zeros(wf_n)
    for k in nums:
        num_idx = np.where(lenP == k)[0]
        array_P_med[num_idx] = np.median(
            np.hstack(np.take(pred_array_P, num_idx)), axis=0
        )
        array_S_med[num_idx] = np.median(
            np.hstack(np.take(pred_array_S, num_idx)), axis=0
        )
        array_M_med[num_idx] = np.median(
            np.hstack(np.take(pred_array_mask, num_idx)), axis=0
        )
    del pred_array_P
    del pred_array_S
    del pred_array_mask
    gc.collect()

    array_P_med = array_P_med[pad_bef:-pad_aft]
    array_S_med = array_S_med[pad_bef:-pad_aft]
    array_M_med = array_M_med[pad_bef:-pad_aft]
    assert len(array_P_med) == wf_npts

    return array_P_med, array_S_med, array_M_med
