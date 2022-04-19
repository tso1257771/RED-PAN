import os
import numpy as np
from scipy.signal import find_peaks
from obspy.signal.trigger import trigger_onset

def picker_info(tar_mask, tar_p, tar_s,
        args = {
            "detection_threshold":0.3, 
            "P_threshold":0.1, 
            "S_threshold":0.1}
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
    p_peaks = find_peaks(tar_p, 
        height=args['P_threshold'], distance=100)[0]
    s_peaks = find_peaks(tar_s, 
        height=args['S_threshold'], distance=100)[0]
    eq_detects = trigger_onset(tar_mask, 
        args['detection_threshold'], args['detection_threshold'])
        
    if len(eq_detects)==0:
        return {}

    eq_detects = eq_detects[
        np.delete(np.arange(len(eq_detects)), 
        np.where((eq_detects[:, 1]-eq_detects[:, 0] < 50))[0])
    ]
    
    eq_collections = {}
    for e in range(len(eq_detects)):
        try:
            potential_P = p_peaks[np.array([p in np.arange(
                eq_detects[e][0]-100, eq_detects[e][1]+100) 
                    for p in p_peaks])]
            potential_S = s_peaks[np.array([s in np.arange(
                eq_detects[e][0]-100, eq_detects[e][1]+100) 
                    for s in s_peaks])]

            P_peak_idx = np.argmax(tar_p[potential_P])
            S_peak_idx = np.argmax(tar_s[potential_S])

            eq_collections[eq_detects[e][0]] = [
                # eq mask info
                eq_detects[e][1], 
                np.mean(tar_mask[eq_detects[e][0]:eq_detects[e][1]]),
                # p peak info
                potential_P[P_peak_idx],
                tar_p[potential_P[P_peak_idx]],
                # s peak info
                potential_S[S_peak_idx],
                tar_s[potential_S[S_peak_idx]]
            ]
        except:
            continue
    return eq_collections


