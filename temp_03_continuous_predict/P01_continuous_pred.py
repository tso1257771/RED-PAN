"""
Seismic phase picking on continuous data
Input: SAC 
Output: SAC
"""
import os
import sys
sys.path.append("../")
sys.path.append("../REDPAN_tools")
import logging
import shutil
import numpy as np
import tensorflow as tf
from glob import glob
from obspy import read
from model_loader import redpan_picker

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_devices = tf.config.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s : %(asctime)s : %(message)s"
)
### Model selection. 
# Please make sure the model (absolute/relative) path is correct.
# assign ${mdl_hdr} with "REDPAN_30s" or "REDPAN_60s"
# REDPAN_30s: model of 30-second-long receptive field
# REDPAN_60s: model of 60-second-long receptive field
mdl_hdr = "REDPAN_60s" # assign with "REDPAN_30s" or "REDPAN_60s"
model_path = f"../pretrained_model/{mdl_hdr}"

### define waveform and output directories
datadir = "./Ridgecrest_WFs"
# find waveform directories
Ddir = np.unique(glob(os.path.join(datadir, "????.???.??")))
outdir = f"./out_data/{mdl_hdr}_pred_Ridgecrest_WFs_post"
# if you want to remove prediction directories, uncommand this line
#shutil.rmtree(outdir, ignore_errors=True)

### prediction parameters and information
# delta, we use broadband seismometers with sample rate of 100 Hz
dt = 0.01
# sliding window length for making predictions
pred_interval_sec = 10
# set `bandpass=None` to disable waveform bandpass filtering
bandpass = [5, 45]
# Post-processing configurations. 
# This only preserve the thresholded values. Others are filled with zeros.
postprocess_config = {
    "mask_trigger": [0.3, 0.3], # trg_on and trg_off threshold
    "mask_len_thre": 0.5, # minimum length of mask in seconds
    "mask_err_win": 0.5, # potential window in seconds for mask error
    "detection_threshold": 0.5, # detection threshold for mask
    "P_threshold": 0.3, # detection threshold for P
    "S_threshold": 0.3 # detection threshold for S
}

### load the encapsulated RED-PAN API
# set postprocess_config=None to disable postprocessing
picker, pred_npts = redpan_picker(
    model_path=model_path, 
    pred_interval_sec=pred_interval_sec,
    dt=dt,
    postprocess_config=postprocess_config
)

### Start making predictions sequentially
# You may want to parallelize this part using multiprocessing.
for D in range(len(Ddir)):
    print(f"Directory: {D+1}/{len(Ddir)}")
    ## waveform index for reading waveform using `obspy.read` class
    ## in the next loop;
    sacs = glob(os.path.join(Ddir[D], "*.sac"))
    wf_idx = np.unique(
        [
            ".".join(os.path.basename(s).split(".")[:3])[:-1]
            + "?."
            + ".".join(os.path.basename(s).split(".")[3:])
            for s in sacs
        ]
    )

    for ct, p in enumerate(wf_idx):
        logging.info(
            f"Processing {os.path.join(Ddir[D], wf_idx[ct])}:"
            f" {ct+1}/{len(wf_idx)} | Directory: {D+1}/{len(Ddir)}")
        
        wf = read(os.path.join(Ddir[D], wf_idx[ct]))
        if bandpass:
            wf = wf.detrend("demean").filter("bandpass", 
                    freqmin=bandpass[0], freqmax=bandpass[1])
            
        # ${STMF_max_sec} is the maximum length of 
        # STMF (Seismogram-Tracking-Median-Filter) in seconds.
        # This restricts the maximum length of applying STMF to
        # prevent from memory insufficiency. Default to 1200.
        
        P_stream, S_stream, M_stream = \
            picker.annotate_stream(wf, STMF_max_sec=1800)
        
        ### write continuous predictions into sac format
        outDdir = os.path.join(outdir, os.path.basename(Ddir[D]))
        if not os.path.exists(outDdir):
            os.makedirs(outDdir)

        W = [P_stream, S_stream, M_stream]
        W_name = wf_idx[ct].replace("?", "")
        W_out = [W_name + ".P", W_name + ".S", W_name + ".mask"]
        for k in range(3):
            out_name = os.path.join(outDdir, W_out[k])
            W[k].write(out_name, format="SAC")
