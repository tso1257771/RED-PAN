import os
import h5py
import sys
sys.path.append('../REDPAN_tools')
import obspy
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from copy import deepcopy
from glob import glob
from obspy.io.sac.sactrace import SACTrace
from obspy import read, UTCDateTime
from generate_data_utils import assign_slice_window
from generate_data_utils import snr_pt_v2, gen_tar_func
from data_utils import PhasePicker, sac_len_complement
from mtan_ARRU import unets
from REDPAN_picker import picker_info
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_devices = tf.config.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.basicConfig(level=logging.INFO,
    format='%(levelname)s : %(asctime)s : %(message)s')
######################################### define basic information
datadir = './'
outdir = os.path.join(datadir, 'wf_data', 'data')
meta_out = os.path.join(datadir, 'wf_data', 'evlist.csv')
#os.system(f'rm -rf {outdir}')
os.remove(f'{outdir}')
if not os.path.exists(outdir):
    os.makedirs(outdir)
meta_f = open(meta_out, 'w')
meta_f.write("source_id\tnetwork\tstation\tloc\tchannel\tpsres\n")

eq_h5 = h5py.File(
    os.path.join(datadir, 'INSTANCE/data/Instance_events_gm_10k.hdf5'), 'r')
eq_meta = pd.read_csv(
    os.path.join(datadir, 'INSTANCE/metadata/metadata_Instance_events_10k.csv'),
     keep_default_na=False, dtype={'station_location_code': object, 
    'source_mt_eval_mode':object,'source_mt_status': object,
    'source_mechanism_strike_dip_rake': object, 
    'source_mechanism_moment_tensor': object, 
    'trace_p_arrival_time': object, 'trace_s_arrival_time': object})

# select machine with most collect waveform 
eq_id = np.array(eq_h5['data'])
machine_ids = np.array(['.'.join(e.split('.')[1:5]) for e in eq_id])
machine_id, m_id_ct = np.unique(machine_ids, return_counts=True)
sel_machine_id = machine_id[np.argmax(m_id_ct)]
sel_meta_idx = (eq_meta.station_network_code+'.'+\
    eq_meta.station_code+'..'+eq_meta.station_channels==sel_machine_id)
sel_eq_meta = eq_meta[sel_meta_idx]

################################## Load REDPAN model
mdl_hdr = 'REDPAN_60s'
pred_npts = 6000
### load self-defined model and data process framework 
model_h5 = os.path.join(f'../pretrained_model/{mdl_hdr}', 'train.hdf5')
# load model and weights
model = unets(input_size=(pred_npts, 3)).build_mtan_R2unet(model_h5)
pick_args = {
    "detection_threshold":0.5, "P_threshold":0.3, "S_threshold":0.3}
### initialize continuous data processing framework
RP_picker = PhasePicker(model=model, pred_npts=pred_npts,
     dt=0.01, postprocess_config=None) 

collect_df = []
ct = 0
for i in range(len(sel_eq_meta)):
    logging.info(f"Generateing sac files with REDPAN predictions: "+\
        f"{i+1}/{len(sel_eq_meta)}")
    eq_info = sel_eq_meta[i:i+1]
    source_id = eq_info.source_id.values[0]
        
    sta_network = eq_info.station_network_code.values[0]
    sta = eq_info.station_code.values[0]
    sta_loc_code = eq_info.station_location_code.values[0]
    sta_chn = eq_info.station_channels.values[0]

    wf_idx = f'{source_id}.{sta_network}.{sta}.{sta_loc_code}.{sta_chn}'
    eq_array = np.array(eq_h5['data'][wf_idx])

    trc_stt = UTCDateTime(eq_info.trace_start_time.values[0])
    trc_dt = eq_info.trace_dt_s.values[0]
    
    if trc_dt != 0.01:
        continue

    # make stream
    tr_ = []
    tr_chn = ['E', 'N', 'Z']
    for j in range(3):
        fake_array = np.hstack([np.zeros(pred_npts), 
            eq_array[j], np.zeros(pred_npts)])
        tr_.append(obspy.Trace(data=fake_array))
        tr_[j].stats.starttime = UTCDateTime(trc_stt) - pred_npts*trc_dt
        tr_[j].stats.delta = trc_dt
        tr_[j].stats.channel = sta_chn+tr_chn[j]
        tr_[j].stats.station = sta
        tr_[j].stats.network = sta_network
    stream = obspy.Stream(tr_)

    _st = deepcopy(stream).detrend('demean').filter(
            'bandpass', freqmin=1, freqmax=45)
    ### phase picking and detection by RED-PAN
    predP, predS, predM = RP_picker.predict(_st, postprocess=False)
    matches = picker_info(predM, predP, predS, pick_args)

    # only consider waveform with single earthquake waveform
    if len(matches) != 1:
        continue

    match_info = matches[list(matches.keys())[0]]
    p_sec, s_sec = match_info[2]*trc_dt, match_info[4]*trc_dt

    outsac_dir = os.path.join(outdir, str(source_id))
    if not os.path.exists(outsac_dir):
        os.makedirs(outsac_dir)
    # fake up sac files
    st_stt = UTCDateTime(trc_stt)
    st_ent = stream[0].stats.endtime - pred_npts*trc_dt
    stream = sac_len_complement(stream.slice(st_stt, st_ent))
    for _sac in stream:
        chn = _sac.stats.channel
        out_sac_temp = os.path.join(outsac_dir, 
            f'{wf_idx}{chn[-1]}.sac')
        _sac.write(out_sac_temp, format='SAC')
    read_idx = out_sac_temp.replace('Z.sac', '?.sac')

    st = read(read_idx)
    new_sac_dict = st[0].stats.sac.copy()
    new_sac_dict['t1'] = p_sec - pred_npts*trc_dt
    new_sac_dict['t2'] = s_sec - pred_npts*trc_dt

    # make new sac
    for s in st:
        s.stats.sac = obspy.core.AttribDict(new_sac_dict)   
        chn = s.stats.channel 
        out_sac = f'{wf_idx}{chn[-1]}.sac'
        #print(os.path.join(outpath, out_norm))
        wf = SACTrace.from_obspy_trace(s)
        wf.b = 0
        wf.write(os.path.join(outsac_dir, out_sac))
    if sta_loc_code == '':
        sta_loc_code = 99
    # write simplified metadata
    info = f"{str(source_id)}\t{sta_network}\t{sta}\t{str(sta_loc_code)}\t"+\
        f"{sta_chn}\t{s_sec-p_sec:.2f}\n"
    meta_f.write(info)
    ct += 1
meta_f.close()

logging.info(f"Generate {ct} sets of single-event waveform in total: {outdir}")
logging.info(f"Created simplified metadata: {meta_out}")
