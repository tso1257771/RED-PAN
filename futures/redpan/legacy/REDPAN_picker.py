import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from obspy.signal.trigger import trigger_onset


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

def extract_picks(raw_wf, P_stream, S_stream, M_stream, dt=0.01, 
        p_amp_estimate_sec=1, s_amp_estimate_sec=3, args={
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

    args : dic
        A dictionary containing all of the input parameters.  
        
    Returns
    -------    
    pick_df : pandas.DataFrame
        Contains the information for the detected and picked event.            
        
    """
    
    trc_id = raw_wf[0].id[:-1]
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

    pick_df = list()
    for K in matches.items():
        if np.any([N==None for N in  [K[1][2], K[1][4]]]):
            continue
        if K[1][4] - K[1][2] < 100:
            continue

        P_pt = init_stt + K[1][2]*dt
        S_pt =  init_stt + K[1][4]*dt

        P_timestamp = P_pt.datetime.isoformat(timespec='milliseconds')
        S_timestamp = S_pt.datetime.isoformat(timespec='milliseconds')
        P_pb = K[1][3]
        S_pb =  K[1][5]
        P_amp = np.max(np.abs(wf_data[:, K[1][2] : K[1][2]+p_amp_estimate_npts ]))
        S_amp = np.max(np.abs(wf_data[:, K[1][4] : K[1][4]+s_amp_estimate_npts ]))

        pick_df.append({
            "id": trc_id,
            "timestamp": P_timestamp,
            "amp": P_amp,
            "prob": P_pb,
            "type": 'p'
        }) 
        pick_df.append({
            "id": trc_id,
            "timestamp": S_timestamp,
            "amp": S_amp,
            "prob": S_pb,
            "type": 's'
        })
    pick_df = pd.DataFrame(pick_df)
    #              id                timestamp           amp      prob type
    # 0    CI.CCC..HH  2019-07-07T07:50:08.637   9198.138504  0.543340    p
    # 1    CI.CCC..HH  2019-07-07T07:50:14.257  13686.979834  0.914223    s
    # 2    CI.CCC..HH  2019-07-07T07:50:48.377   2612.497834  0.399723    p
    # 3    CI.CCC..HH  2019-07-07T07:50:53.147   4540.355622  0.796342    s
    # 4    CI.CCC..HH  2019-07-07T07:52:23.767   2129.819425  0.552528    p
    # ..          ...                      ...           ...       ...  ... 
    return pick_df