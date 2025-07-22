import os
import csv
import sys
sys.path.append('../REDPAN_tools')
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 14
from glob import glob
from obspy import read, UTCDateTime
from REDPAN_picker import picker_info

mdl = 'REDPAN_60s'
predir = f'./out_data/{mdl}_pred_Ridgecrest_WFs'
outf = f'./out_data/pick_{mdl}_Ridgecrest_WFs.csv'
daydir = np.sort(glob(os.path.join(predir, '????.???.??')))
buffer_sec = 600
dt = 0.01
pick_args = {
    "detection_threshold":0.3, 
    "P_threshold":0.1, 
    "S_threshold":0.1}

row_id = "sta, dayid, trigger_on_utc, trigger_off, mask_pb,"+\
        " p_pt, p_pb, s_pt, s_pb"
f = open(outf, 'w')
print(row_id, file=f)

for ct, p in enumerate(daydir):
    print(f'{ct+1}/{len(daydir)}: {daydir[ct]}')
    day_id = os.path.basename(daydir[ct])
    sta_info = np.sort(np.unique(['.'.join(s.split('/')[-1].split('.')[:3]) 
        for s in glob(os.path.join(daydir[ct], '*'))]))
    for s in range(len(sta_info)):
        _trc = read(os.path.join(daydir[ct], f'{sta_info[s]}.*.P'), 
            header_only=True)[0].stats
        init_stt = _trc.starttime + buffer_sec
        init_end = _trc.endtime - buffer_sec

        predP = read(os.path.join(daydir[ct], f'{sta_info[s]}.*.P')
            ).slice(init_stt, init_end)[0].data
        predS = read(os.path.join(daydir[ct], f'{sta_info[s]}.*.S')
            ).slice(init_stt, init_end)[0].data
        predM = read(os.path.join(daydir[ct], f'{sta_info[s]}.*.mask')
            ).slice(init_stt, init_end)[0].data
                
        matches = picker_info(predM, predP, predS, pick_args)

        K_ct = 0
        for K in matches.items():
            if np.any([N==None for N in  [K[1][2], K[1][4]]]):
                continue
            detect_st_pt = init_stt + K[0]*dt
            detect_ed_pt = init_stt + K[1][0]*dt
            P_pt = init_stt + K[1][2]*dt
            S_pt =  init_stt + K[1][4]*dt
            detect_pb =  K[1][1]
            P_pb = K[1][3]
            S_pb =  K[1][5]
            K_ct += 1
            #print(detect_pb, P_pb, S_pb, K_ct)

            message = \
                f"{sta_info[s].split('.')[1]}\t{day_id}\t{detect_st_pt}\t"+\
                f"{detect_ed_pt-detect_st_pt:4.2f}\t{detect_pb:4.2f}\t"+\
                f"{P_pt-detect_st_pt:4.2f}\t{P_pb:4.2f}\t"+\
                f"{S_pt-detect_st_pt:4.2f}\t{S_pb:4.2f}"
            print(message, file=f)

        if matches == {}:
            continue
f.close()
    
