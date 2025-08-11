import os, gc, sys, shutil, psutil, logging
sys.path.append("../../")
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
from glob import glob
from obspy import read, UTCDateTime
from redpan import inference_engine
from redpan.models import unets
from redpan.utils import sac_len_complement
from redpan.picks import extract_picks

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_devices = tf.config.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s : %(asctime)s : %(message)s"
)

def log_memory_usage(stage):
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"Memory usage at {stage}: {memory_mb:.1f} MB")

# fill in 'REDPAN_30s' or 'REDPAN_60s' for using model of different receptive field
mdl_hdr = 'REDPAN_60s_240107' 

### define waveform and output directories    
datadir = 'outputs/SAC'
outdir = 'outputs/redpan_picks'

### load self-defined model and data process framework 
model_h5 = os.path.join(f'../../pretrained_model/{mdl_hdr}', 'train.hdf5')

### prediction parameters and information
# delta, we use broadband seismometers with sample rate of 100 Hz 
dt = 0.01  
# data length of model input
pred_npts = 6000 
# sliding window length for making predictions
pred_interval_sec = 10
# set `bandpass=None` to disable waveform bandpass filtering
bandpass = [3, 45] 

# load model and weights
frame = unets()
model = frame.build_mtan_R2unet(model_h5, input_size=(pred_npts, 3))

# define post-processing parameters if needed
postprocess_config = {
    'mask_trigger': [0.1, 0.1], 
    'mask_len_thre': 0.5,
    'mask_err_win':0.5, 
    'trigger_thre':0.3
}

# the threshold value for EQ detection mask, P/S arrival picking
pick_args = {
    "detection_threshold":0.3, 
    "P_threshold":0.3, 
    "S_threshold":0.1
}

### initialize continuous data processing framework
# you could specify `postprocess_config=postprocess_config``
# to enable prediction postprocessing
picker = inference_engine(
    model=model, 
    pred_interval_sec=pred_interval_sec,
    dt=dt,
    batch_size=16,
    postprocess_config=None
)

try:
    I = int(sys.argv[1])
    process_n = int(sys.argv[2])
except:
    I = 0
    process_n = 1

# Define the directory list for searching sac files
Ddir = np.unique(glob(os.path.join(datadir, '2016.???')))

for D in range(len(Ddir)):
    print(f"Directory: {D+1}/{len(Ddir)}: {Ddir[D]}")
    ## waveform index for reading waveform using `obspy.read` class
    ## in the next loop;
    sacs = glob(os.path.join(Ddir[D], '*.sac'))
    
    [yr, julday] = [int(os.path.basename(Ddir[D]).split('.')[_])  for _ in range(2)]
    ## Define the time frame for RED-PAN inferencing
    # start time: 1 minute prior to the beginning of the day
    slice_stt = UTCDateTime(year=yr, julday=julday) - 1*60
    # end time: 3 minutes after to the end of the day
    slice_ent = UTCDateTime(year=yr, julday=julday) + 24*60*60 + 3*60

    def get_wf_idx(sacs):
        """ Generate waveform index for reading waveforms."""
        wf_idx = []
        for sac in sacs:
            bname = os.path.basename(sac)
            net, sta, loc, chn, _ = bname.split('.')
            glob_idx = f"{net}.{sta}.{loc}.{chn[:-1]}?.sac"
            wf_idx.append(glob_idx)
        wf_idx = np.unique(wf_idx)
        return wf_idx

    wf_idx = get_wf_idx(sacs)

    # Let the order of the wf_idx be the same across different process
    np.random.seed(8844)
    wf_idx = np.random.permutation(wf_idx)
    wf_idx = wf_idx[I::process_n]

    for ct, p in enumerate(wf_idx):
        logging.info(f"Processing {os.path.join(Ddir[D], wf_idx[ct])}:"
              f" {ct+1}/{len(wf_idx)} | Directory: {D+1}/{len(Ddir)}")
        _net, _sta, _loc, _chn = wf_idx[ct].split('.')[:4]
        wfid = f"{_net}.{_sta}.{_loc}.{_chn.replace('?', '')}"

        # check if processed
        yr, julday = [_ for _ in os.path.basename(Ddir[D]).split('.')]
        outDdir = os.path.join(outdir, yr, julday)
        if not os.path.exists(outDdir):
            os.makedirs(outDdir, exist_ok=True)
        # skip if output file exists        
        out_file = os.path.join(outDdir, f'picks_{wfid}.csv')
        if os.path.exists(out_file):
            logging.info(f"{out_file} file exists, skip.")
            continue
        
        # read and slice the waveform
        try:
            wf = read(os.path.join(Ddir[D], wf_idx[ct]))
        except:
            logging.error(f"Failed to read waveform {wf_idx[ct]}")
            continue

        if len(wf) != 3:
            for _ in range(3-len(wf)):
                wf.append(wf[-1])
        wf = wf.slice(slice_stt, slice_ent)

        if bandpass:
            wf = wf.detrend('demean').filter('bandpass', 
                freqmin=bandpass[0], freqmax=bandpass[1])

        ### complement sac data when the length is not consistent across channels
        wf = sac_len_complement(wf)
        if len(wf[0].data) < pred_npts:
            continue

        ################################################################
        # RED-PAN phase picking
        ################################################################
        try:
            # Inference
            t1 = UTCDateTime()
            P_stream, S_stream, M_stream = picker.annotate_stream(
                wf, postprocess=None)
            t2 = UTCDateTime()
            print(f"Time taken for prediction: {t2 - t1}")
        except AttributeError:
            continue
        # Extract picks
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
            args={"detection_threshold":0.3, "P_threshold":0.3, "S_threshold":0.1}
        )
        # drop amplitude columns since we do not remove the response
        pick_df = pick_df.drop(columns={'amp'})
        pick_df.to_csv(out_file, index=False)
        logging.info(f"Saved picks to {out_file}")

        # del st, P_stream, S_stream, M_stream, pick_df
        # # tf.keras.backend.clear_session()
        # gc.collect()

        log_memory_usage("after processing")
