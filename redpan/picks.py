import os
import gc
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.signal import find_peaks
from obspy.signal.trigger import trigger_onset
from obspy import UTCDateTime

def picker_info(
    tar_mask,
    tar_p,
    tar_s,
    args={"detection_threshold": 0.3, "P_threshold": 0.1, "S_threshold": 0.1},
):
    """ 
    Performs detection and picking on predicted data.
    Parameters
    ----------
    mask : 1D array
        Detection probabilities. 
        
    tar_p : 1D array
        P arrival probabilities.  
        
    tar_s : 1D array
        S arrival probabilities. 

    args : dic
        A dictionary containing all of the input parameters.  
    Returns
    -------    
    matches : dic
        Contains the information for the detected and picked event.            
        
    matches : dic
        {detection start-time:
        [ detection end-time, detection probability, 
        P arrival, P probabiliy, 
        S arrival,  S probability]}
    """
    p_peaks = find_peaks(tar_p, height=args["P_threshold"], distance=100)[0]
    s_peaks = find_peaks(tar_s, height=args["S_threshold"], distance=100)[0]
    eq_detects = trigger_onset(
        tar_mask, args["detection_threshold"], args["detection_threshold"]
    )

    if len(eq_detects) == 0:
        return {}

    eq_detects = eq_detects[
        np.delete(
            np.arange(len(eq_detects)),
            np.where((eq_detects[:, 1] - eq_detects[:, 0] < 50))[0],
        )
    ]

    eq_collections = {}
    for e in range(len(eq_detects)):
        try:
            potential_P = p_peaks[
                np.array(
                    [
                        p in np.arange(eq_detects[e][0] - 100, eq_detects[e][1] + 100)
                        for p in p_peaks
                    ]
                )
            ]
            potential_S = s_peaks[
                np.array(
                    [
                        s in np.arange(eq_detects[e][0] - 100, eq_detects[e][1] + 100)
                        for s in s_peaks
                    ]
                )
            ]

            P_peak_idx = np.argmax(tar_p[potential_P])
            S_peak_idx = np.argmax(tar_s[potential_S])

            eq_collections[eq_detects[e][0]] = [
                # eq mask info
                eq_detects[e][1],
                np.mean(tar_mask[eq_detects[e][0] : eq_detects[e][1]]),
                # p peak info
                potential_P[P_peak_idx],
                tar_p[potential_P[P_peak_idx]],
                # s peak info
                potential_S[S_peak_idx],
                tar_s[potential_S[S_peak_idx]],
            ]
        except:
            continue
    return eq_collections

def extract_picks(raw_wf, P_stream, S_stream, M_stream, 
        station_id=None, dt=0.01, 
        p_amp_estimate_sec=1, s_amp_estimate_sec=3, 
        starttime=None, endtime=None, args={
        "detection_threshold":0.5, "P_threshold":0.3, "S_threshold":0.3
        }
    ):
    """ 
    Performs detection and picking on predicted data.
    Parameters
    ----------
    raw_wf : obspy.core.stream.Stream
        Stream containing the raw data for amplitude estimation purpose. 
        
    P_stream : obspy.core.stream.Stream
        Stream containing the predicted P arrival times.  
        
    S_stream : obspy.core.stream.Stream
        Stream containing the predicted S arrival times. 

    M_stream : obspy.core.stream.Stream
        Stream containing the predicted mask. 

    dt : float
        Trace delta.

    p_amp_estimate_sec : float
        Time span in seconds for estimating P amplitude.
        
    s_amp_estimate_sec : float
        Time span in seconds for estimating S amplitude.
    starttime : str or UTCDateTime or None
        Start time for filtering the P picks. If None, no filtering is applied.
    endtime : str or UTCDateTime or None
        End time for filtering the P picks. If None, no filtering is applied.
    args : dic
        A dictionary containing all of the input parameters.  
        
    Returns
    -------    
    pick_df : pandas.DataFrame
        Contains the information for the detected and picked event.            
        
    """
    if not station_id:
        station_id = raw_wf[0].id[:-1]
    init_stt = raw_wf[0].stats.starttime
    wf_data = np.array([W.data for W in raw_wf])
    p_amp_estimate_npts = int(p_amp_estimate_sec/dt)
    s_amp_estimate_npts = int(s_amp_estimate_sec/dt)
    matches = picker_info(
        M_stream[0].data, 
        P_stream[0].data, 
        S_stream[0].data, 
        args = args
    )

    K_ct = 0
    matches_keys = np.array(list(matches.keys()))
    message_dfs = list()
    for k in range(len(matches_keys)):
        K = matches[matches_keys[k]]
        if np.any([N==None for N in  [K[2], K[4]]]):
            continue
        if K[4] - K[2] < 100:
            continue

        P_pt = init_stt + K[2]*dt
        S_pt =  init_stt + K[4]*dt

        # Neglect if the P arrival is later than $endtime
        if endtime and P_pt > UTCDateTime(endtime):
            continue
        if starttime and P_pt < UTCDateTime(starttime):
            continue
        P_timestamp = P_pt.datetime.isoformat(timespec='milliseconds')
        S_timestamp = S_pt.datetime.isoformat(timespec='milliseconds')
        P_pb = np.round(K[3], 2)
        S_pb =  np.round(K[5], 2)

        P_amp = np.max(np.abs(wf_data[:, K[2] : K[2]+p_amp_estimate_npts ]))
        S_amp = np.max(np.abs(wf_data[:, K[4] : K[4]+s_amp_estimate_npts ]))

        message_df = pd.DataFrame(
            [[station_id, P_timestamp, P_amp, P_pb, 'p', K_ct],
                [station_id, S_timestamp, S_amp, S_pb, 's', K_ct]], 
        )
        message_df.columns = ['id', 'timestamp', 'amp', 'prob', 'type', 'pick_idx']
        message_dfs.append(message_df)
        K_ct += 1
        
    if len(message_dfs) == 0:
        print(f"No pair for {station_id} after checking phase-time orders")
        message_df_all = pd.DataFrame(
            columns=['id', 'timestamp', 'amp', 'prob', 'type', 'pick_idx'])
        return message_df_all         
    elif len(message_dfs) > 1:
        message_df_all = pd.concat(message_dfs).reset_index(drop=True)
    elif len(message_dfs) == 1:
        message_df_all = message_dfs[0]

    message_df_all['timestamp'] = message_df_all['timestamp'].apply(UTCDateTime)
    # The P arrival of latter pair should not be earlier than the S arrival of former pair
    msg_df_p = message_df_all[message_df_all.type=='p']
    msg_df_s = message_df_all[message_df_all.type=='s']
    keep_pick_idx = np.where(
        (msg_df_p.timestamp.values[1:] - msg_df_s.timestamp.values[:-1]) > 0)
    if len(keep_pick_idx) == 0:
        print(f"No pair for {station_id} after checking phase-time orders")
        message_df_all = pd.DataFrame(
            columns=['id', 'timestamp', 'amp', 'prob', 'type', 'pick_idx'])
        return message_df_all
    message_df_all = message_df_all[message_df_all.pick_idx.apply(
        lambda x: x in np.hstack([[0], keep_pick_idx[0]+1]))]
    assert len(message_df_all)%2 == 0
    message_df_all.loc[:, 'pick_idx'] = message_df_all.index//2
    message_df_all.loc[:, 'timestamp'] = message_df_all.timestamp.apply(
        lambda x: x.datetime.isoformat(timespec='milliseconds'))
    #             id                timestamp           amp  prob type  pick_idx
    # 0   TW.TPUB..BH  2024-04-04T00:04:38.239  7.078234e-09  0.52    p         0
    # 1   TW.TPUB..BH  2024-04-04T00:04:56.599  6.432185e-08  0.72    s         0
    # 2   TW.TPUB..BH  2024-04-04T00:07:38.579  8.427526e-09  0.51    p         1
    # 3   TW.TPUB..BH  2024-04-04T00:07:54.849  1.535819e-08  0.44    s         1
    # 4   TW.TPUB..BH  2024-04-04T00:10:11.639  1.520155e-08  0.53    p         2
    # 5   TW.TPUB..BH  2024-04-04T00:10:30.029  2.956276e-08  0.44    s         2
    # ..          ...                      ...           ...       ...  ... 
    return message_df_all

def IsRightTrigger(gt_range, trigger_mask):
    """
    gt_range: trigger range of ground truth
    trigger_mask: trigger ranges of prediction masks
    """
    # 1 for right Trigger; 0 for wrong Trigger/No trigger
    def check_trigger(gt_range, trg_range):
        if trg_range[0] > gt_range[0] and trg_range[1] < gt_range[1]:
            return 1
        else:
            return 0

    T = np.sum([check_trigger(gt_range, t) for t in trigger_mask])
    if T > 1:
        T = 0
    return T


def trg_peak_value(pred_func, trg_st_thre=0.1, trg_end_thre=0.1):
    """
    Check the maximum value of predictions trigger function
    """
    trg_func = trigger_onset(pred_func, trg_st_thre, trg_end_thre)
    if len(trg_func) == 0:
        max_pk_value = 0
    else:
        peak_values = []
        for trg in trg_func:
            peak_value = np.max(pred_func[trg])
            if peak_value >= 0.1:
                peak_values.append(peak_value)
        max_pk_value = np.max(peak_values)
    return max_pk_value


def pick_peaks(
    prediction, labeled_phase, sac_dt=None, search_win=1, peak_value_min=0.01
):
    """
    search for potential pick
    
    parameters
    ----
    prediction: predicted functions
    labeled_phase: the timing of labeled phase
    sac_dt: delta of sac 
    search_win: time window (sec) for searching 
    local maximum near labeled phases 
    """
    try:
        tphase = int(round(labeled_phase / sac_dt))
        search_range = [
            tphase - int(search_win / sac_dt),
            tphase + int(search_win / sac_dt),
        ]
        peaks, values = find_peaks(prediction, height=peak_value_min)

        in_search = [
            np.logical_and(v > search_range[0], v < search_range[1]) for v in peaks
        ]
        _peaks = peaks[in_search]
        _values = values["peak_heights"][in_search]
        return _peaks[np.argmax(_values)] * sac_dt, _values[np.argmax(_values)]
    except ValueError:
        return -999, -999

def pick_peaks_from_predictions(
    prediction, labeled_phase_sec, sac_dt=None, search_win=1, peak_value_min=0.01
):
    """
    search for potential pick
    
    parameters
    ----
    prediction: predicted functions
    labeled_phase: the timing of labeled phase
    sac_dt: delta of sac 
    search_win: time window (sec) for searching 
    local maximum near labeled phases 
    """
    try:
        tphase = int(round(labeled_phase_sec / sac_dt))
        search_range = [
            tphase - int(search_win / sac_dt),
            tphase + int(search_win / sac_dt),
        ]
        peaks, values = find_peaks(prediction, height=peak_value_min)

        in_search = [
            np.logical_and(v > search_range[0], v < search_range[1]) for v in peaks
        ]
        _peaks = peaks[in_search]
        _values = values["peak_heights"][in_search]
        return _peaks[np.argmax(_values)] * sac_dt, _values[np.argmax(_values)]
    except ValueError:
        return None, None
        

def pred_postprocess(
    array_P_med,
    array_S_med,
    array_M_med,
    dt=0.01,
    mask_trigger=[0.1, 0.1],
    mask_len_thre=0.5,
    mask_err_win=0.5,
    detection_threshold=0.3,
    P_threshold=0.1,
    S_threshold=0.1
):
    """Predictions postprocessing
    1. filter the prediction functions using detection mask
    2. zero-padding non-detected space
    """

    pick_args = {
        "detection_threshold": detection_threshold, 
        "P_threshold": P_threshold,
        "S_threshold": S_threshold
    }

    n_len = len(array_P_med)
    mask_min_len = int(mask_len_thre / dt)
    err_win = int(mask_err_win / dt)

    matches = picker_info(array_M_med, array_P_med, array_S_med, pick_args)

    r_funM, r_funP, r_funS = np.zeros(n_len), np.zeros(n_len), np.zeros(n_len)

    for K in matches.items():
        if np.any([N == None for N in [K[1][2], K[1][4]]]):
            continue
        detect_st_pt = K[0]
        detect_ed_pt = K[1][0]
        P_pt = K[1][2]
        S_pt = K[1][4]

        r_funM[detect_st_pt:detect_ed_pt] = \
            deepcopy(array_M_med[detect_st_pt:detect_ed_pt])
        r_funP[detect_st_pt-err_win : detect_ed_pt+err_win] = \
            deepcopy(array_P_med[detect_st_pt-err_win :detect_ed_pt+err_win])
        r_funS[detect_st_pt-err_win : detect_ed_pt+err_win] = \
            deepcopy(array_S_med[detect_st_pt-err_win : detect_ed_pt+err_win])

    return r_funP, r_funS, r_funM