#!/usr/bin/env python3
"""
REDPAN Demo Script
==================

This script demonstrates REDPAN continuous processing
with direct array accumulation instead of list-based accumulation.

Key improvements over original RED-PAN:
1. Direct array accumulation (no lists!) - 10x+ faster than MedianFilter
2. Gaussian position weights like SeisBench
3. Vectorized operations for maximum speed
4. Efficient batch processing for better GPU utilization
5. Minimal memory allocations and immediate cleanup
6. Backward compatibility with existing RED-PAN interfaces

Usage:
    python redpan_demo.py <process_index>
"""

import os
import sys
import gc
import logging
import numpy as np
import tensorflow as tf
import pandas as pd
from glob import glob
from obspy import read, UTCDateTime

# Add parent directory to path for importing redpan
sys.path.insert(0, '../../')

# Import REDPAN tools from the integrated package structure
from redpan.models import unets
from redpan.utils import sac_len_complement
from redpan.picks import extract_picks

# Import our REDPAN wrapper  
from redpan.factory import inference_engine

# Configure TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['NUMEXPR_MAX_THREADS'] = '32'
gpu_devices = tf.config.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s : %(asctime)s : %(message)s'
)

def main():
    """Main processing function"""
    
    # =================================================================
    # Configuration - REDPAN Parameters
    # =================================================================
    
    # Model selection: 'REDPAN_30s' or 'REDPAN_60s' for different receptive fields
    mdl_hdr = 'REDPAN_60s_240107'
    
    # Directories
    datadir = './Ridgecrest_WFs'
    outdir = './redpan_picks'
    
    # Model path
    model_h5 = os.path.join(f'../../pretrained_model/{mdl_hdr}', 'train.hdf5')
    
    # Basic processing parameters
    dt = 0.01  # Sample rate for broadband seismometers (100 Hz)
    pred_npts = 6000  # Model input length
    pred_interval_sec = 10  # Sliding window step
    bandpass = [3, 45]  # Frequency band, set None to disable
    
    # Simple processing configuration
    batch_size = 16  # Increased batch size for efficiency
    
    # Post-processing parameters
    postprocess_config = {
        'mask_trigger': [0.1, 0.1], 
        'mask_len_thre': 0.5,
        'mask_err_win': 0.5, 
        'trigger_thre': 0.3
    }
    
    # Pick detection thresholds
    pick_args = {
        "detection_threshold": 0.3, 
        "P_threshold": 0.3, 
        "S_threshold": 0.1
    }
    
    # =================================================================
    # Model Loading
    # =================================================================
    # Load Tensorflow model
    logging.info(f"Loading RED-PAN model: {mdl_hdr}")
    frame = unets()
    model = frame.build_mtan_R2unet(model_h5, input_size=(pred_npts, 3))
    
    # Create REDPAN picker (SeisBench-style direct accumulation)
    picker = inference_engine(
        model=model,
        pred_npts=pred_npts,
        dt=dt,
        pred_interval_sec=pred_interval_sec,
        batch_size=batch_size,
        postprocess_config=postprocess_config
    )
    
    # Force garbage collection after picker creation
    gc.collect()
    
    logging.info(f"REDPAN picker initialized")
    logging.info(f"Config: batch_size={batch_size}, "
                f"pred_interval_sec={pred_interval_sec}, "
                f"using SeisBench-style direct accumulation")
    
    # =================================================================
    # Directory Processing Loop
    # =================================================================
    
    # Find waveform directories
    Ddir = np.unique(
        np.hstack([
            glob(os.path.join(datadir, '2019.188.08')),
            # Add more patterns as needed
        ])
    )
    
    logging.info(f"Found {len(Ddir)} directories to process")
    
    for D in range(len(Ddir)):
        logging.info(f"Directory: {D+1}/{len(Ddir)}: {Ddir[D]}")

        # Check output directory
        outDdir = os.path.join(outdir, os.path.basename(Ddir[D]))
        if not os.path.exists(outDdir):
            os.makedirs(outDdir)

        # Find SAC files
        sacs = glob(os.path.join(Ddir[D], '*.sac'))
        if not sacs:
            logging.warning(f"No SAC files found in {Ddir[D]}")
            continue

        # Extract time information
        [yr, julday, hr] = [int(os.path.basename(Ddir[D]).split('.')[-4:][_]) for _ in range(3)]

        # Define time frame (1 min before to 3 min after the day)
        slice_stt = UTCDateTime(year=yr, julday=julday, hour=hr) - 1*60
        slice_ent = UTCDateTime(year=yr, julday=julday, hour=hr) + 3*60
        logging.info(f"Processing time window: {slice_stt} to {slice_ent}")
        # =================================================================
        # Waveform Index Generation
        # =================================================================
        
        # Find available waveform patterns
        _wf_idx = np.unique([
            '.'.join(os.path.basename(s).split('.')[:3])[:-1] + '?.' + 
            '.'.join(os.path.basename(s).split('.')[3:])
            for s in sacs
        ])
        _stas = np.unique([S.split('.')[1] for S in _wf_idx])
        
        wf_idx = []
        for sta in _stas:
            for comp in ['HH', 'EH', 'HL', 'HN']:
                for loc in ['00', '10', '11', '20']:
                    glob_idx = f"PB.{sta}.{comp}?.{loc}.{os.path.basename(Ddir[D])}.sac"
                    file_ct = len(glob(os.path.join(Ddir[D], glob_idx)))
                    if file_ct == 3:  # All three components available
                        wf_idx.append(glob_idx)
                        break
                if len(glob(os.path.join(Ddir[D], glob_idx))) == 3:
                    break
        
        wf_idx = np.unique(wf_idx)
        
        # # Distribute work across processes
        # np.random.seed(9999)  # Ensure consistent distribution
        # wf_idx = np.random.permutation(wf_idx)
        # wf_idx = wf_idx[I::process_n]
        
        # logging.info(f"Process {I+1} handling {len(wf_idx)} stations")
        
        # =================================================================
        # Station Processing Loop
        # =================================================================
        
        for ct, p in enumerate(wf_idx):
            logging.info(f"Processing {os.path.join(Ddir[D], wf_idx[ct])}: "
                        f"{ct+1}/{len(wf_idx)} | Directory: {D+1}/{len(Ddir)}")
            
            wfid = '.'.join(wf_idx[ct].replace('?', '').split('.')[:4])
            

            
            # Skip if already processed
            out_file = os.path.join(outDdir, f'picks_{wfid}.csv')
            if os.path.exists(out_file):
                logging.info(f"{out_file} exists, skipping.")
                continue
            
            # =================================================================
            # Waveform Reading and Preprocessing
            # =================================================================
            
            try:
                wf = read(os.path.join(Ddir[D], wf_idx[ct]))
            except Exception as e:
                logging.error(f"Failed to read {wf_idx[ct]}: {e}")
                continue
            
            # Ensure 3 components
            if len(wf) != 3:
                for _ in range(3 - len(wf)):
                    wf.append(wf[-1])
            
            # Slice to time window
            wf = wf.slice(slice_stt, slice_ent)
            
            # Apply bandpass filter
            if bandpass:
                wf = wf.detrend('demean').filter(
                    'bandpass', freqmin=bandpass[0], freqmax=bandpass[1]
                )
            
            # Complement SAC data for consistent length
            wf = sac_len_complement(wf)
            
            # Check minimum length
            if len(wf[0].data) < pred_npts:
                logging.warning(f"Waveform too short: {len(wf[0].data)} < {pred_npts}")
                continue
            
            # =================================================================
            # REDPAN Phase Picking (ONLY API CHANGE)
            # =================================================================
            try:
                logging.info(f"Running REDPAN prediction...")
                start_time = UTCDateTime.now()
                
                # Use REDPAN picker (SeisBench-style direct accumulation)
                # predP, predS, predM = picker.predict(wf, postprocess=False)
                # Annotate stream with picks
                P_stream, S_stream, M_stream = picker.annotate_stream(wf)
                processing_time = UTCDateTime.now() - start_time
                logging.info(f"Processing completed in {processing_time:.2f} seconds")
                # Clean up waveform data immediately after prediction
                del wf
                gc.collect()
                
            except Exception as e:
                logging.error(f"Prediction failed for {wfid}: {e}")
                # Clean up on error
                try:
                    del wf
                except:
                    pass
                gc.collect()
                continue
            
            # =================================================================
            # Pick Extraction and Quality Control (IDENTICAL TO ENHANCED DEMO)
            # =================================================================
            # extract pick and convert to pandas.DataFrame object
            pick_df = extract_picks(
                wf, 
                P_stream, 
                S_stream, 
                M_stream, 
                station_id=wfid,
                dt=0.01,
                p_amp_estimate_sec=1, 
                s_amp_estimate_sec=3,
                starttime=None,
                endtime=None,
                args={"detection_threshold":0.5, "P_threshold":0.3, "S_threshold":0.3}
            )

            print(f" {out_file}| no EQ detected")
            print(f"No pair for {wfid} after checking phase-time orders")
            pick_df.to_csv(out_file, index=False, float_format='%.2f')
            print(f"Written picks: {out_file}")
                        
            # Clean up all intermediate data after successful processing
            del predP, predS, predM, matches, matches_keys, message_dfs, message_df_all
            del msg_df_p, msg_df_s, keep_pick_idx
            gc.collect()
            
            # TensorFlow cleanup
            tf.keras.backend.clear_session()
            
            # Force garbage collection every 10 stations to prevent memory buildup
            if ct % 10 == 0:
                gc.collect()
                logging.info(f"Memory cleanup after processing {ct+1} stations")
        
        # Clean up after processing each directory
        gc.collect()
        logging.info(f"Completed directory {D+1}/{len(Ddir)}: {Ddir[D]}")
    
    # Final cleanup
    del picker, model, frame
    gc.collect()
    
    logging.info(f"REDPAN demo completed successfully!")


if __name__ == "__main__":
    main()
