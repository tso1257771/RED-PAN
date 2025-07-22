import os
import sys
sys.path.append('../REDPAN_tools')
import obspy
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import pandas as pd
from copy import deepcopy
from glob import glob
from obspy import read, UTCDateTime
from generate_data_utils import assign_slice_window
from generate_data_utils import snr_pt_v2, gen_tar_func
from data_io import write_TFRecord_detect
logging.basicConfig(level=logging.INFO,
            format='%(levelname)s : %(asctime)s : %(message)s')

# defined waveform path
wfdir = './wf_data/data'
meta_df = pd.read_table('./wf_data/evlist.csv', header=0, delimiter='\s+',
    names=['source_id', 'network', 'station', 'loc', 'channel', 'psres'])
# define output path
outpath = './tfrecord'
outdir_single = os.path.join(outpath, 'singleEQ')
figdir_single = os.path.join(outpath, 'fig/fig_singleEQ')

#os.system(f'rm -rf {outdir_single} {figdir_single}')
os.remove(f'{outdir_single} {figdir_single}')
os.makedirs(outdir_single); os.makedirs(figdir_single)

wf_npts = 6000
err_win_p, err_win_s = 0.4, 0.6 #half_win*2
gen_num = len(meta_df)

# generate data 
gen_ct = 0
gen_iter = 0
for j in range(len(meta_df)):
    if gen_ct <= gen_num:
        logging.info(f"Generating SingleEQ data: {gen_ct+1}/{gen_num}")
        info = meta_df[j:j+1].values[0]
        gen_iter += 1
    else:
        logging.info("Finished generating SingleEQ data.")
        logging.info(outdir_single)
        logging.info(outdir_Ponly)
        break

    wf = read(os.path.join(wfdir, str(info[0]),
        f'{str(info[0])}.{info[1]}.{info[2]}.*.{info[4]}?.sac'))

    # t1 and t2 are picked by our pretrained RED-PAN model
    wf.sort()
    hdr = wf[0].stats.sac
    ## generate regular single-earthquake waveform
    tp_utc = wf[0].stats.starttime - hdr.b + hdr.t1
    ts_utc = wf[0].stats.starttime - hdr.b + hdr.t2

    tp_npts = int((tp_utc-wf[0].stats.starttime)*100)
    ts_npts = int((ts_utc-wf[0].stats.starttime)*100)

    if np.logical_or(
        len(wf[0].data[tp_npts-300:tp_npts+300]) < 500,
        len(wf[0].data[ts_npts-300:ts_npts+300]) < 500):
        #stop
        continue
    
    wf_init_stt = tp_utc - 0.5
    if (ts_utc - tp_utc) < 20:
        wf_init_ent = wf_init_stt + 0.5 + 2*(ts_utc - tp_utc)
    elif 20 <= (ts_utc - tp_utc) < 40:
        wf_init_ent = wf_init_stt + 0.5 + 1.4*(ts_utc - tp_utc)
    elif 40 <= (ts_utc - tp_utc):
        wf_init_ent = wf_init_stt + 0.5 + 0.4*(ts_utc - tp_utc)

    available_npts_total = wf_npts - int((wf_init_ent-wf_init_stt)/0.01)
    available_npts_bef = int((wf_init_stt-wf[0].stats.starttime)/0.01)

    slice_stt = wf_init_stt - np.random.randint(available_npts_bef)*0.01
    # slice waveform
    slice_stt_npts = int((slice_stt - wf[0].stats.starttime)*100)
    slice_tp_npts = int((tp_utc - slice_stt)*100)
    slice_ts_npts = int((ts_utc - slice_stt)*100)
    if np.logical_or(slice_tp_npts>= wf_npts, slice_ts_npts>= wf_npts):
        continue

    slice_trc = []
    for s in range(3):
        _trc = wf[s].data[slice_stt_npts:slice_stt_npts+wf_npts]
        _trc -= np.mean(_trc)
        _trc /= np.std(_trc)
        if np.any(np.isnan(_trc)):
            _trc[np.isnan(_trc)] = 0
        if np.any(np.isinf(_trc)):
            _trc[np.isinf(_trc)] = 0
        slice_trc.append(_trc)  
    slice_trc = np.array(slice_trc)

    # phase picking target function
    trc_tp = gen_tar_func(wf_npts, slice_tp_npts, 
        int(np.round(err_win_p/0.01)))
    trc_ts = gen_tar_func(wf_npts, slice_ts_npts, 
        int(np.round(err_win_s/0.01)))
    trc_tn = np.ones(wf_npts) - trc_tp - trc_ts

    # mask target function
    trc_mask = trc_tp + trc_ts
    trc_mask[slice_tp_npts:slice_ts_npts+1] = 1
    trc_unmask = np.ones(wf_npts) - trc_mask

    # reshape for input U net model
    trc_3C = np.array([slice_trc[0], slice_trc[1], slice_trc[2]]).T
    label_psn = np.array([trc_tp, trc_ts, trc_tn]).T
    mask = np.array([trc_mask, trc_unmask]).T

    if np.logical_or(np.isinf(trc_3C).any(), 
            np.isnan(trc_3C).any()):
        raise ValueError
    
    ev_idx = f'{str(info[0])}.{info[1]}.{info[2]}.{info[4]}'
    idx = f'{ev_idx}_{gen_ct+1:07}'
    outfile = os.path.join(outdir_single, f'{idx}.tfrecord')

    if gen_ct <= 30:
        fig, ax = plt.subplots(6, 1, figsize=(10, 8))
        ax[0].plot(slice_trc[0], linewidth=1)
        ax[1].plot(slice_trc[1], linewidth=1)
        ax[2].plot(slice_trc[2], linewidth=1)
        ax[0].set_ylabel('E'); ax[1].set_ylabel('N'); ax[2].set_ylabel('Z')
        ax[3].plot(trc_tp, linewidth=1); ax[3].set_ylim(0, 1.1)
        ax[3].axvline(slice_tp_npts, linewidth=1, linestyle=':')
        ax[4].axvline(slice_ts_npts, linewidth=1, linestyle=':')
        ax[4].plot(trc_ts, linewidth=1); ax[4].set_ylim(0, 1.1)
        ax[5].plot(trc_mask, linewidth=1); ax[5].set_ylim(0, 1.1)
        ax[3].set_ylabel('P'); ax[4].set_ylabel('S'); ax[5].set_ylabel('mask')
        ax[5].set_xlabel('npts')
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(figdir_single, f'{idx}.png'))
        plt.close()

    write_TFRecord_detect(trc_3C, label_psn, mask, 
                idx=idx, outfile=outfile)
    gen_ct += 1


