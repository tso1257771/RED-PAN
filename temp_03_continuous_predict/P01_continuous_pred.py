"""
Seismic phase picking on continuous data
Input: SAC 
Output: SAC
"""
import os
import sys
sys.path.append('../')
import logging
import numpy as np
import tensorflow as tf
from glob import glob
from obspy import read
from REDPAN_tools.mtan_ARRU import unets
from REDPAN_tools.data_utils import PhasePicker
from REDPAN_tools.data_utils import sac_len_complement
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_devices = tf.config.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.basicConfig(level=logging.INFO,
    format='%(levelname)s : %(asctime)s : %(message)s')

mdl_hdr = 'REDPAN_60s'
### define waveform and output directories    
datadir = './Ridgecrest_WFs'
outdir = f'./out_data/{mdl_hdr}_pred_Ridgecrest_WFs'

# if you want to remove prediction directories, uncommand this line
os.system(f'rm -rf {outdir}')

### load self-defined model and data process framework 
model_h5 = os.path.join(f'../pretrained_model/{mdl_hdr}', 'train.hdf5')

### prediction parameters and information
# delta, we use broadband seismometers with sample rate of 100 Hz 
dt = 0.01  
# data length of model input
pred_npts = 6000 
# sliding window length for making predictions
pred_interval_sec = 4
# set `bandpass=None` to disable waveform bandpass filtering
bandpass = [1, 45] 

# load model and weights
frame = unets()
model = frame.build_mtan_R2unet(model_h5, 
        input_size=(pred_npts, 3))

# define post-processing parameters if needed
postprocess_config={
    'mask_trigger': [0.1, 0.1], 
    'mask_len_thre': 0.5,
    'mask_err_win':0.5, 
    'trigger_thre':0.3
    }

### initialize continuous data processing framework
# you could specify `postprocess_config=postprocess_config``
# to enable prediction postprocessing
picker = PhasePicker(model=model, pred_npts=pred_npts,
     dt=dt, postprocess_config=None) 

## find waveform directories
Ddir = np.unique(glob(os.path.join(datadir, '????.???.??')))
for D in range(len(Ddir)):
    print(f"Directory: {D+1}/{len(Ddir)}")
    ## waveform index for reading waveform using `obspy.read` class
    ## in the next loop;
    sacs = glob(os.path.join(Ddir[D], '*.sac'))
    wf_idx = np.unique([ '.'.join(
        os.path.basename(s).split('.')[:3])[:-1]+'?.'+\
        '.'.join(os.path.basename(s).split('.')[3:])
        for s in sacs])

    for ct, p in enumerate(wf_idx):
        logging.info(f"Processing {os.path.join(Ddir[D], wf_idx[ct])}:"
              f" {ct+1}/{len(wf_idx)} | Directory: {D+1}/{len(Ddir)}")
        wf = read(os.path.join(Ddir[D], wf_idx[ct]))

        if bandpass:
            wf = wf.detrend('demean').filter('bandpass', 
                freqmin=bandpass[0], freqmax=bandpass[1])

        ### complement sac data when the length is not consistent across channels
        wf = sac_len_complement(wf)
        ### phase picking and detection
        array_P_med, array_S_med, array_M_med = picker.predict(
            wf, postprocess=False)
            
        ### check if any infinity or nan (not a number) exists
        assert np.any(np.isinf(array_P_med))==False
        assert np.any(np.isnan(array_P_med))==False

        ### write continuous predictions into sac format
        outDdir = os.path.join(outdir, os.path.basename(Ddir[D]))
        if not os.path.exists(outDdir): 
            os.makedirs(outDdir)

        W_name =  wf_idx[ct].replace('?', '')
        W_out = [W_name+'.P', W_name+'.S', W_name+'.mask']
        W_data = [array_P_med, array_S_med, array_M_med]
        W_chn = ['P', 'S', 'mask']
        for k in range(3):
            out_name = os.path.join(outDdir, W_out[k])
            W = wf[0].copy()
            W.data = W_data[k]
            W.stats.channel = W_chn[k]
            W.write(out_name, format='SAC')
