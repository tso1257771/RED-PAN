import os
import sys
sys.path.append('../R')
sys.path.append('../REDPAN_tools')
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import scipy.signal as ss
from glob import glob
from obspy import read
from sklearn.utils import shuffle
from data_io import write_TFRecord_detect
from data_aug import mosaic_wf_plot_detect
from data_aug import mosaic_basic_info
from data_aug import mosaic_wf_marching
from data_aug import mosaic_relative_amp
from data_aug import mosaic_tar_func_detect
os.environ["CUDA_VISIBLE_DEVICES"] = ""
logging.basicConfig(level=logging.INFO,
        format='%(levelname)s : %(asctime)s : %(message)s')

wfdir = './wf_data/data'
meta_df = pd.read_table('./wf_data/evlist.csv', header=0, delimiter='\s+',
    names=['source_id', 'network', 'station', 'loc', 'channel', 'psres'])
# define output path
outdir = os.path.join('./tfrecord', 'EEWA')
figdir = os.path.join('./tfrecord', 'fig/fig_EEWA')
#os.system(f'rm -rf {outdir} {figdir}')
os.remove(f'{outdir} {figdir}')
os.makedirs(outdir); os.makedirs(figdir)
# mosaic waveform information
out_f = os.path.join(outdir, 'info.txt')
f = open(out_f, 'w')
f.write(f'file;tp;ts;max_amp_chn;relative_amp\n')

total_gen_n = 30

base_wf_sec = 60
data_npts = 6000
ev_num = [2, 3, 4]
buffer_secs_bef_P = 0.5
buffer_EEWA_secs_bef_P = 3
dt = 0.01
suffix_Psnr_thre = 1.5
joint_snr_thre = 1.5
# least eq waveform defined by |ts-tp|  
psres_multiple = 1.5 # times
err_win_p, err_win_s = 0.4, 0.6
marching_range = (3, 15)

df_machine_types = meta_df.station+'.'+meta_df.channel
machine_types, m_ct = np.unique(df_machine_types, return_counts=True)
ct_max = int(total_gen_n/len(ev_num))
cts = []
for e in range(len(ev_num)):
    ct = 0
    plot_ct = 0
    mosaic_n = ev_num[e]
    max_eqlen = base_wf_sec / mosaic_n

    while ct <= ct_max:
        for i in range(len(machine_types)):
            machine_type = machine_types[i]
            mosaic_df = meta_df[df_machine_types == machine_type]
            # 0.5 refers to buffer of P phase
            mosaic_df = mosaic_df[ 
                mosaic_df.psres*psres_multiple + buffer_secs_bef_P < max_eqlen]
            
            #if len(mosaic_df) < 300:
            #    continue
            
            mosaic_df = shuffle(mosaic_df)
            mosaic_info = mosaic_df.values

            # use same signal or not
            same_wf_TRUE = np.random.random()
            if same_wf_TRUE < 0.3:
                use_info = np.vstack([mosaic_info[0] for m in range(mosaic_n)])
            else:
                use_info = mosaic_info[:mosaic_n]
            # 'year' 'jday', 'evid', 'station', 'channel', 
            # 'psres', 'matched', 'EQ_detected'

            # formatted as text that could be parsed by obspy.read
            read_path_info = np.array([os.path.join(wfdir, str(u[0]),
                 f'{u[0]}.{u[1]}.{u[2]}.*.{u[4]}?.sac')
                for u in use_info])
                
            # residual estimate from |S-P| residual
            p_utc, s_utc, trc_pairs, init_stt, init_ent, base_period = \
                mosaic_basic_info(read_path_info=read_path_info, 
                    psres_multiple=psres_multiple,
                    base_wf_sec=base_wf_sec, 
                    buffer_secs_bef_P=buffer_secs_bef_P, dt=dt)

            if base_period == None:
                continue

            if not np.array_equal(
                np.array([len(i) for i in trc_pairs]), 
                np.full((len(trc_pairs)), 3)):
                continue

            trc_mosaic_pair, tp_to_ori_pt_pair, ts_to_ori_pt_pair =\
                mosaic_wf_marching(base_wf_sec=base_wf_sec, dt=dt,
                    base_period=base_period, 
                    data_npts=data_npts, 
                    p_utc=p_utc, s_utc=s_utc, 
                    trc_pairs=trc_pairs, 
                    init_stt=init_stt, 
                    init_ent=init_ent,
                    marching=True, 
                    marching_range=marching_range,
                    joint_snr_thre=joint_snr_thre,
                    suffix_Psnr_thre=suffix_Psnr_thre,
                    scaling=False,
                    EEW_march=True,
                    buffer_EEWA_secs_bef_P=buffer_EEWA_secs_bef_P)

            if type(trc_mosaic_pair) == type(None):
                continue

            # make mosaic waveform and target functions
            # (with marching window)
            for sp in range(len(trc_mosaic_pair)):
                # remove disappeared points after marching 
                valid_tp_to_ori_pt = tp_to_ori_pt_pair[sp][
                    np.logical_and( 
                        tp_to_ori_pt_pair[sp] > 0,
                        tp_to_ori_pt_pair[sp] < data_npts)
                    ]

                valid_ts_to_ori_pt = ts_to_ori_pt_pair[sp][
                    np.logical_and( 
                        ts_to_ori_pt_pair[sp] > 0,
                        ts_to_ori_pt_pair[sp] < data_npts)
                    ]    

                # estimate relative amplitude
                max_amp_chn, relative_mag = mosaic_relative_amp(
                    trc_mosaic=trc_mosaic_pair[sp], 
                    tp_to_ori_pt=valid_tp_to_ori_pt,
                    hp_freq=2, dt=dt, data_npts=data_npts)

                label_psn, mask = mosaic_tar_func_detect(
                    data_npts=data_npts, 
                    err_win_p=err_win_p, err_win_s=err_win_s, dt=dt, 
                    tp_to_ori_pt=valid_tp_to_ori_pt, 
                    ts_to_ori_pt=valid_ts_to_ori_pt)

                trc_3C = trc_mosaic_pair[sp].T
                assert trc_3C.shape == label_psn.shape == (data_npts, 3)

                # make mosaic waveform
                outid = f'EEWA_{ev_num[e]}evs_{machine_type}.{ct+1:07}'
                outfile = os.path.join(outdir, f'{outid}.tfrecord')
                logging.info(f'Generating mosaic waveform ({ev_num[e]}evs) - '
                    f'{ct+1}/{ct_max} \n {outfile}')                    
                write_TFRecord_detect(trc_3C, label_psn, mask,
                            idx=outid, outfile=outfile)
                # write arrivals information
                tp_secs = list(np.round(valid_tp_to_ori_pt*dt, 2))
                ts_secs = list(np.round(valid_ts_to_ori_pt*dt, 2))
                out_info = f'{outfile};{tp_secs};{ts_secs}'+\
                            f';{max_amp_chn};{list(relative_mag)}'
                print(out_info, file=f)

                ct += 1
                wf_data = trc_3C.T
                tar_func = label_psn.T

                if plot_ct < 100:
                    savefig = os.path.join(figdir, f'{outid}.png')
                    fig = mosaic_wf_plot_detect(
                            trc_3C, label_psn, mask,
                            valid_tp_to_ori_pt, valid_ts_to_ori_pt, 
                            save=savefig, show=False)
                    plot_ct += 1
