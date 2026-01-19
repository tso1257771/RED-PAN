import os
import logging
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Helvetica"
from obspy import read
from obspy.signal import filter
from scipy.signal import tukey
from glob import glob
from redpan.data.generate_data_utils import gen_tar_func
from redpan.data.generate_data_utils import sac_len_complement
from redpan.data.generate_data_utils import check_snr_for_waveform
from redpan.data.data_utils import MWA_joint_ENZsnr, MWA_suffix_Psnr
from redpan.utils import generate_matching_noise, find_reference_signal

logger = logging.getLogger(__name__)


def mosaic_basic_info(
    read_path_info, psres_multiple=1.5, base_wf_sec=60, buffer_secs_bef_P=0.5, dt=0.01
):
    """
    return basic information for making mosaic waveform
    """
    p_utc = []
    s_utc = []
    residual_ps = []
    trc_pairs = []
    for t in range(len(read_path_info)):
        try:
            trc = sac_len_complement(read(read_path_info[t]))
            if len(trc) != 3:
                lack_n = 3 - len(trc)
                for N in range(lack_n):
                    trc.append(trc[-1])
        except:
            raise ValueError("Check the path of waveform !")
            continue
        info = trc[0].stats
        tp = info.starttime - info.sac.b + info.sac.t1
        ts = info.starttime - info.sac.b + info.sac.t2

        residual = ts - tp

        p_utc.append(tp)
        s_utc.append(ts)
        residual_ps.append(residual)
        trc_pairs.append(trc)

    if len(trc_pairs) != len(read_path_info):
        return None, None, None, None, None, None

    p_utc = np.array(p_utc)
    s_utc = np.array(s_utc)

    residual_ps = np.array(residual_ps)

    init_stt = np.array(p_utc) - buffer_secs_bef_P
    init_ent = np.array(p_utc) + psres_multiple * np.array(residual_ps)
    base_period = np.sum(init_ent - init_stt)

    if base_period >= base_wf_sec:
        print("Noooooo!!!")
        return None, None, None, None, None, None
    else:
        return p_utc, s_utc, trc_pairs, init_stt, init_ent, base_period


def mosaic_basic_info_instance(
    read_path_info, psres_multiple=1.5, base_wf_sec=60, buffer_secs_bef_P=0.5, dt=0.01
):
    """
    INSTANCE-specific version: reads t1/t2 for P/S picks instead of using SAC 'b' offset.
    Return basic information for making mosaic waveform.
    
    INSTANCE SAC headers:
    - t1: REDPAN P pick (relative to trace start)
    - t2: REDPAN S pick (relative to trace start)
    - t3: Manual P pick (may be -12345 if not set)
    - t4: Manual S pick (may be -12345 if not set)
    
    Args:
        read_path_info: List of file path patterns to read waveforms
        psres_multiple: Multiplier for P-S residual to determine window end
        base_wf_sec: Base waveform length in seconds
        buffer_secs_bef_P: Buffer time before P arrival
        dt: Sample interval in seconds
        
    Returns:
        Tuple of (p_utc, s_utc, trc_pairs, init_stt, init_ent, base_period)
        or (None, None, None, None, None, None) if invalid
    """
    p_utc = []
    s_utc = []
    residual_ps = []
    trc_pairs = []
    
    for t in range(len(read_path_info)):
        try:
            trc = read(read_path_info[t])
            trc.sort()
            if len(trc) != 3:
                lack_n = 3 - len(trc)
                for N in range(lack_n):
                    trc.append(trc[-1].copy())
        except Exception as err:
            logger.debug(f"Failed to read {read_path_info[t]}: {err}")
            continue
            
        info = trc[0].stats
        hdr = info.sac
        
        # INSTANCE uses t1/t2 for REDPAN picks (primary)
        # Use manual picks t3/t4 if available and valid
        pred_tp = hdr.t1  # REDPAN P
        pred_ts = hdr.t2  # REDPAN S
        
        man_tp = getattr(hdr, 't3', -12345.0)
        man_ts = getattr(hdr, 't4', -12345.0)
        
        # Use manual if valid, otherwise use REDPAN
        if man_tp is not None and man_tp != -12345.0:
            tp_pick = man_tp
        else:
            tp_pick = pred_tp
            
        if man_ts is not None and man_ts != -12345.0:
            ts_pick = man_ts
        else:
            ts_pick = pred_ts
        
        # Calculate absolute times
        # For INSTANCE data: t1/t2 are relative to trace starttime
        tp = info.starttime + tp_pick
        ts = info.starttime + ts_pick
        
        residual = ts - tp
        if residual <= 0:
            logger.debug(f"Invalid P-S residual: {residual}")
            continue
            
        p_utc.append(tp)
        s_utc.append(ts)
        residual_ps.append(residual)
        trc_pairs.append(trc)
        
    if len(trc_pairs) != len(read_path_info):
        return None, None, None, None, None, None

    p_utc = np.array(p_utc)
    s_utc = np.array(s_utc)
    residual_ps = np.array(residual_ps)

    init_stt = np.array(p_utc) - buffer_secs_bef_P
    init_ent = np.array(p_utc) + psres_multiple * np.array(residual_ps)
    base_period = np.sum(init_ent - init_stt)

    if base_period >= base_wf_sec:
        logger.debug(f'Base period {base_period} >= {base_wf_sec}')
        return None, None, None, None, None, None
    else:
        return p_utc, s_utc, trc_pairs, init_stt, init_ent, base_period


def fill_flat_regions_in_mosaic(trc_mosaic, window_size=100, min_unique=50):
    """
    Fill flat regions in the final mosaic waveform with spectrum-matched noise.
    
    Flat regions (constant or near-constant values) can occur from data gaps,
    clipping, or instrument issues. This function replaces them with realistic
    noise that matches the spectral characteristics of the surrounding signal.
    
    Args:
        trc_mosaic: Mosaic waveform array with shape (3, data_npts) for E, N, Z
        window_size: Size of window to check for flat regions (default: 100)
        min_unique: Minimum unique values to consider non-flat (default: 50)
        
    Returns:
        Array with same shape as input, with flat regions filled with noise
    """
    trc_filled = trc_mosaic.copy()
    
    for ch in range(3):
        data = trc_filled[ch].copy()
        
        # Find reference signal for this channel
        reference_signal = find_reference_signal(
            data, window_size=300, max_search=len(data)-300, min_unique=200
        )
        
        # Fill flat regions
        for j in range(0, len(data) - window_size, window_size):
            segment = data[j:j+window_size]
            n_unique = len(np.unique(np.round(segment, decimals=4)))
            if n_unique < min_unique:
                if reference_signal is not None and len(reference_signal) > 50:
                    noise_fill = generate_matching_noise(reference_signal, window_size)
                    data[j:j+window_size] = noise_fill
                else:
                    # Fallback: add small random noise
                    data[j:j+window_size] = np.random.normal(0, 1e-6, window_size)
        
        trc_filled[ch] = data
    
    return trc_filled


def mosaic_wf_marching(
    base_wf_sec,
    dt,
    base_period,
    data_npts,
    p_utc,
    s_utc,
    trc_pairs,
    init_stt,
    init_ent,
    marching=True,
    marching_range=(1, 7),
    joint_snr_thre=3,
    suffix_Psnr_thre=1.5,
    scaling=False,
    EEW_march=False,
    buffer_EEWA_secs_bef_P=3,
):

    available_pts = np.round(data_npts - base_period / dt).astype("int")
    buffer_for_mosaic = 0.1
    # if available_pts < 0:
    #    continue
    if available_pts < 0:
        print(available_pts)

    assert available_pts > 0
    # appended after every mosaic and before the first mosaic
    p_s_residual = s_utc - p_utc
    assign_res = (
        np.random.multinomial(
            available_pts, np.random.dirichlet(np.ones(len(trc_pairs) + 1))
        )
        * dt
    )
    ## set up starttime and endtime point for each trace
    ## sum up to base_wf_sec
    slice_idx_center = []
    for ass in range(len(trc_pairs)):
        slice_idx_center.append([init_stt[ass], init_ent[ass] + assign_res[ass + 1]])
        if ass == 0:
            slice_idx_center[ass][0] = init_stt[ass] - assign_res[ass]
    slice_idx_center = np.array(slice_idx_center)

    # bandpass if the trc paris are the same
    if len(np.unique([t[0].stats.starttime for t in trc_pairs])) != 1:
        for T in trc_pairs:
            T = T.detrend("demean")
            T = T.filter("bandpass", freqmin=1, freqmax=45)
            T = T.taper(max_percentage=0.05)

    # choose random window for shifting
    if not EEW_march:
        shift_for_sec, shift_back_sec = (
            np.random.randint(marching_range[0], marching_range[1]),
            np.random.randint(marching_range[0], marching_range[1]),
        )

    elif EEW_march:
        try:
            shift_for_sec, shift_back_sec = (
                np.random.randint(
                    assign_res[0] / dt + buffer_EEWA_secs_bef_P,
                    (p_s_residual[0] + assign_res[0]) / dt,
                )
                * dt,
                np.random.randint(
                    (p_s_residual[-1] * 0.5 + assign_res[-1]) / dt,
                    (p_s_residual[-1] * 1.5 + assign_res[-1] - 0.5) / dt,
                )
                * dt,
            )
        except:
            return None, None, None

    ### fake up base trace
    # 1. calculate every mosaic point by utc time and slice range
    acc = []
    mosaic_utc = []
    slice_idx_to_end = []
    total_sec = base_wf_sec + shift_for_sec + shift_back_sec + buffer_for_mosaic

    _mosaic_utc_ori = slice_idx_center[0][1]
    for s in range(len(slice_idx_center)):
        if s == 0:
            ext_stt = slice_idx_center[s][0] - shift_back_sec
        else:
            ext_stt = slice_idx_center[s][0]
        ext_ent = ext_stt + (total_sec - np.sum(acc))

        if s != len(slice_idx_center) - 1:
            mosaic_utc.append(_mosaic_utc_ori + np.sum(acc))
            acc.append(slice_idx_center[s][1] - ext_stt)
            _mosaic_utc_ori += np.sum(acc)

        slice_idx_to_end.append([ext_stt, ext_ent])
    cumsum_npts = np.round(np.cumsum(acc) / dt).astype(int)

    # 2. stack waveform and estimate accordingly new p/s utc time
    new_p_utc = []
    new_s_utc = []
    new_mosaic_utc = []

    if slice_idx_to_end[0][1] - trc_pairs[0][0].stats.endtime >= 0:
        print("raw sac file not long enough !")
        return None, None, None

    base_trc = sac_len_complement(
        trc_pairs[0].slice(slice_idx_to_end[0][0], slice_idx_to_end[0][1])
    )

    # if scaling:
    #    scale = np.random.uniform(scale[0], scale[1])
    #    for b in base_trc:
    #        b.data = b.data*scale

    base_trc.sort()
    base_trc.detrend("demean")
    base_trc_n = len(base_trc[0])
    for ct, p in enumerate(slice_idx_to_end):
        if ct == 0:
            new_p_utc.append(p_utc[ct])
            new_s_utc.append(s_utc[ct])
        else:
            if p[1] - trc_pairs[ct][0].stats.endtime > 0:
                return None, None, None

            mosaic_trc = sac_len_complement(trc_pairs[ct].slice(p[0], p[1]))

            mosaic_pt = cumsum_npts[ct - 1]

            new_p_utc.append(
                slice_idx_to_end[0][0] + mosaic_pt * dt + (p_utc[ct] - p[0])
            )
            new_s_utc.append(
                slice_idx_to_end[0][0] + mosaic_pt * dt + (s_utc[ct] - p[0])
            )

            mosaic_trc.sort()
            mosaic_trc = mosaic_trc.detrend("demean")
            mosaic_ENZ = np.array([i.data for i in mosaic_trc])

            # stack waveform
            mosaic_n = np.min([len(i) for i in mosaic_ENZ])
            avail_npts = np.min([len(i.data[mosaic_pt:]) for i in base_trc])

            for m in range(len(mosaic_ENZ)):
                if scaling:
                    scale = np.random.uniform(scaling[0], scaling[1])
                elif not scaling:
                    scale = 1
                if mosaic_n >= avail_npts:
                    base_trc[m].data[mosaic_pt : mosaic_pt + avail_npts] += (
                        mosaic_ENZ[m][:avail_npts] * scale
                    )
                else:
                    base_trc[m].data[mosaic_pt : mosaic_pt + mosaic_n] += (
                        mosaic_ENZ[m][:mosaic_n] * scale
                    )
            new_mosaic_utc.append(slice_idx_to_end[0][0] + mosaic_pt * dt)

    # forward, center, and backward mosaic waveform
    mosaic_stt = np.array(
        [
            slice_idx_center[0][0] + shift_for_sec,
            slice_idx_center[0][0],
            base_trc[0].stats.starttime,
        ]
    )

    # 3. seperate marching mosaic waveform from base_trc
    joint_pt_pair = []
    tp_to_ori_pt_pair = []
    ts_to_ori_pt_pair = []
    trc_mosaic_pair = []
    for ms in range(len(mosaic_stt)):
        ms_stt = mosaic_stt[ms]
        ms_ent = mosaic_stt[ms] + base_wf_sec + buffer_for_mosaic

        # mark tp/ts/mosaic joint point
        tp_to_ori_sec = np.array(new_p_utc) - mosaic_stt[ms]
        ts_to_ori_sec = np.array(new_s_utc) - mosaic_stt[ms]
        joint_sec = np.array(new_mosaic_utc) - mosaic_stt[ms]
        tp_to_ori_pt = [np.round(i / dt).astype(int) for i in tp_to_ori_sec]
        ts_to_ori_pt = [np.round(i / dt).astype(int) for i in ts_to_ori_sec]
        joint_pt = [np.round(i / dt).astype(int) for i in joint_sec]
        tp_to_ori_pt_pair.append(tp_to_ori_pt)
        ts_to_ori_pt_pair.append(ts_to_ori_pt)
        joint_pt_pair.append(joint_pt)

        # fill in mosaic waveform
        ms_wf = sac_len_complement(base_trc.copy().slice(ms_stt, ms_ent))
        ms_wf.sort()
        fake_E = ms_wf[0].data[:data_npts]
        fake_N = ms_wf[1].data[:data_npts]
        fake_Z = ms_wf[2].data[:data_npts]

        fake = [fake_E, fake_N, fake_Z]
        # Z-score normalization
        for s3 in range(len(fake)):
            fake[s3] = fake[s3] - np.mean(fake[s3])
            fake[s3] = fake[s3] / np.std(fake[s3])
            fake[s3][np.isinf(fake[s3])] = 1e-4
            fake[s3][np.isnan(fake[s3])] = 1e-4
        trc_mosaic = np.array(fake)
        trc_mosaic_pair.append(trc_mosaic)

    # check joint SNR is small enough
    joint_ENZsnrs_pairs = MWA_joint_ENZsnr(
        trc_mosaic_pair[1], joint_pt_pair[1], snr_win=0.5, dt=0.01, hpfreq=2, mode="std"
    )
    # check P SNRs is large enough
    suffix_Psnrs = MWA_suffix_Psnr(
        trc_mosaic_pair[1],
        tp_to_ori_pt_pair[1],
        snr_win=3,
        dt=0.01,
        hpfreq=2,
        mode="std",
    )

    if np.logical_or(
        np.any(np.hstack(joint_ENZsnrs_pairs) > joint_snr_thre),
        np.any(np.hstack(suffix_Psnrs) < suffix_Psnr_thre),
    ):
        # print("SNR tooo small !!!!!!!")
        # print(joint_ENZsnrs_pairs)
        return None, None, None

    if not np.array_equal(
        (3, data_npts), np.unique([i.shape for i in trc_mosaic_pair])
    ):
        return None, None, None

    # np.array(trc_mosaic_pair)[np.isnan(np.array(trc_mosaic_pair))]= 1e-4
    # np.array(trc_mosaic_pair)[np.isinf(np.array(trc_mosaic_pair))]= 1e-4
    if marching == True:
        return (
            np.array(trc_mosaic_pair),
            np.array(tp_to_ori_pt_pair),
            np.array(ts_to_ori_pt_pair),
        )
    else:
        return (
            np.array(trc_mosaic_pair[1]),
            np.array(tp_to_ori_pt_pair[1]),
            np.array(ts_to_ori_pt_pair[1]),
        )


def mosaic_relative_amp(trc_mosaic, tp_to_ori_pt, hp_freq, dt, data_npts):
    if len(tp_to_ori_pt) < 1:
        return "None", [-1]
    amp_maxs = []
    comp = ["E", "N", "Z"]
    hp_mosaic = tukey(data_npts, alpha=0.1) * (
        filter.highpass(trc_mosaic, freq=hp_freq, df=1 / dt)
    )

    tp_to_ori_pt[tp_to_ori_pt < 0] = 0
    for i in range(len(tp_to_ori_pt)):
        tps = tp_to_ori_pt
        if i != len(tp_to_ori_pt) - 1:
            mosaic_piece = hp_mosaic[:, tps[i] : tps[i + 1]]

        else:
            mosaic_piece = hp_mosaic[:, tps[i] :]
        amp_3C = np.max(np.fabs(mosaic_piece), axis=1)
        amp_maxs.append(amp_3C)
    # choose channel with maximum relative amplitude
    try:
        norm_amp = np.array(amp_maxs) / np.min(amp_maxs, axis=0)
        chose_trc, chose_chn = np.where(norm_amp == np.max(norm_amp))
        return comp[chose_chn[0]], norm_amp[:, chose_chn[0]]
    except:
        return "None", [0]


def mosaic_tar_func(data_npts, err_win_p, err_win_s, dt, tp_to_ori_pt, ts_to_ori_pt):
    ## gen_tar_func
    tar_p_data = np.zeros(data_npts)
    tar_s_data = np.zeros(data_npts)

    err_win_npts_p = int(err_win_p / dt) + 1
    err_win_npts_s = int(err_win_s / dt) + 1

    for n in range(len(tp_to_ori_pt)):
        tar_p_data += gen_tar_func(data_npts, tp_to_ori_pt[n], err_win_npts_p)
        tar_s_data += gen_tar_func(data_npts, ts_to_ori_pt[n], err_win_npts_s)
    tar_nz_data = np.ones(data_npts) - tar_p_data - tar_s_data
    label_psn = np.array([tar_p_data, tar_s_data, tar_nz_data]).T
    return label_psn


def mosaic_tar_func_detect(
    data_npts, err_win_p, err_win_s, dt, tp_to_ori_pt, ts_to_ori_pt, label="gaussian"
):
    if label == "gaussian":
        tar_function = gen_tar_func

    ## gen_tar_func
    tar_p_data = np.zeros(data_npts)
    tar_s_data = np.zeros(data_npts)

    err_win_npts_p = int(err_win_p / dt) + 1
    err_win_npts_s = int(err_win_s / dt) + 1

    for n in range(len(tp_to_ori_pt)):
        tar_p_data += tar_function(data_npts, tp_to_ori_pt[n], err_win_npts_p)

    for m in range(len(ts_to_ori_pt)):
        tar_s_data += tar_function(data_npts, ts_to_ori_pt[m], err_win_npts_s)

    tar_nz_data = np.ones(data_npts) - tar_p_data - tar_s_data
    label_psn = np.array([tar_p_data, tar_s_data, tar_nz_data]).T

    trc_mask = tar_p_data + tar_s_data
    # conditions of same amount of labeled tp and ts
    if len(tp_to_ori_pt) == len(ts_to_ori_pt) > 1:
        if np.min(ts_to_ori_pt) < np.min(tp_to_ori_pt):
            trc_mask[: int(ts_to_ori_pt[0]) + 1] = 1
            trc_mask[int(tp_to_ori_pt[-1]) :] = 1
            for arr_pt in range(len(tp_to_ori_pt) - 1):
                trc_mask[
                    int(tp_to_ori_pt[arr_pt]) : int(ts_to_ori_pt[arr_pt + 1]) + 1
                ] = 1
        else:
            for arr_pt in range(len(tp_to_ori_pt)):
                trc_mask[int(tp_to_ori_pt[arr_pt]) : int(ts_to_ori_pt[arr_pt]) + 1] = 1

    elif len(tp_to_ori_pt) == len(ts_to_ori_pt) == 1:
        if ts_to_ori_pt[0] < tp_to_ori_pt[0]:
            trc_mask[: ts_to_ori_pt[0]] = 1
            trc_mask[tp_to_ori_pt[0] :] = 1
        else:
            trc_mask[int(tp_to_ori_pt[0]) : int(ts_to_ori_pt[0]) + 1] = 1

    # labeled tp more than ts
    elif len(tp_to_ori_pt) > len(ts_to_ori_pt):
        trc_mask[int(tp_to_ori_pt[-1] + 1) :] = 1
        for arr_pt in range(len(ts_to_ori_pt)):
            trc_mask[int(tp_to_ori_pt[arr_pt]) : int(ts_to_ori_pt[arr_pt]) + 1] = 1
    # labeled tp less than ts
    elif len(tp_to_ori_pt) < len(ts_to_ori_pt):
        trc_mask[: int(ts_to_ori_pt[0] + 1)] = 1
        for arr_pt in range(len(tp_to_ori_pt)):
            trc_mask[int(tp_to_ori_pt[arr_pt]) : int(ts_to_ori_pt[arr_pt + 1]) + 1] = 1
    trc_unmask = np.ones(data_npts) - trc_mask
    mask = np.array([trc_mask, trc_unmask]).T

    return label_psn, mask


def mosaic_wf_plot(trc_3C, label_psn, new_tps, new_tss, save=None, show=False):
    wf_data = trc_3C.T
    tar_func = label_psn.T

    x_plot = [wf_data[0], wf_data[1], wf_data[2], tar_func[0], tar_func[1], tar_func[2]]
    label = ["E comp.", "N comp.", "Z comp.", "P prob.", "S prob.", "Noise prob"]
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
        6, 1, sharex=True, figsize=(8, 8)
    )
    ax = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(6):
        ax[i].plot(x_plot[i], linewidth=1)
        if 2 >= i:
            for j in range(len(new_tps)):
                ax[i].axvline(
                    x=new_tps[j], label="Manual picked P", color="k", linewidth=1
                )
            for j in range(len(new_tss)):
                ax[i].axvline(
                    x=new_tss[j], label="Manual picked S", color="r", linewidth=1
                )
        ax[i].set_ylabel(label[i])
    ax[0].set_title("Mosaic-concatenated waveform")
    if save:
        plt.savefig(save, dpi=150)
        plt.close()
    if show:
        plt.show()
    return fig


def mosaic_wf_plot_detect(
    trc_3C, label_psn, mask, new_tps, new_tss, save=None, show=False
):
    wf_data = trc_3C.T
    tar_func = label_psn.T
    mask_func = mask.T

    x_plot = [
        wf_data[0],
        wf_data[1],
        wf_data[2],
        tar_func[0],
        tar_func[1],
        mask_func[0],
    ]
    label = ["E comp.", "N comp.", "Z comp.", "P prob.", "S prob.", "Mask prob"]
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
        6, 1, sharex=True, figsize=(8, 8)
    )
    ax = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(6):
        ax[i].plot(x_plot[i], linewidth=1)
        if 2 >= i:
            for j in range(len(new_tps)):
                ax[i].axvline(
                    x=new_tps[j], label="Manual picked P", color="k", linewidth=1
                )
            for j in range(len(new_tss)):
                ax[i].axvline(
                    x=new_tss[j], label="Manual picked S", color="r", linewidth=1
                )
        ax[i].set_ylabel(label[i])
    ax[0].set_title("Mosaic-concatenated waveform")
    if save:
        plt.savefig(save, dpi=150)
        plt.close()
    if show:
        plt.show()
    return fig
