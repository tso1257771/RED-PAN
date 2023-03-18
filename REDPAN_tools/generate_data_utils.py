import obspy
import numpy as np
import scipy.signal as ss
from scipy.signal import tukey
from scipy.signal import find_peaks
from obspy import read, UTCDateTime
from obspy.signal import filter
from obspy.signal.filter import envelope
from obspy.signal.trigger import trigger_onset
from copy import deepcopy


def gen_tar_func(data_length, point, mask_window):
    """
    data_length: target function length
    point: point of phase arrival
    mask_window: length of mask, must be even number
                 (mask_window//2+1+mask_window//2)
    """
    target = np.zeros(data_length)
    half_win = mask_window // 2
    gaus = np.exp(
        -((np.arange(-half_win, half_win + 1)) ** 2) / (2 * (half_win // 2) ** 2)
    )
    # print(gaus.std())
    gaus_first_half = gaus[: mask_window // 2]
    gaus_second_half = gaus[mask_window // 2 + 1 :]
    target[point] = gaus.max()
    # print(gaus.max())
    if point < half_win:
        reduce_pts = half_win - point
        start_pt = 0
        gaus_first_half = gaus_first_half[reduce_pts:]
    else:
        start_pt = point - half_win
    target[start_pt:point] = gaus_first_half
    target[point + 1 : point + half_win + 1] = gaus_second_half[
        : len(target[point + 1 : point + half_win + 1])
    ]

    return target


def assign_slice_window(p_s_residual, data_length, avail_bef_P, avail_aft_S, dt):
    """
    `p_s_residual`: |P_arrival - S_arrival|
    `data_length`: total length of sliced waveform
    `avail_bef_P`: available dataspace before P phase
    `avail_aft_S`: available dataspace ater S phase

    Conditioning
    -P_prewin >= avail_bef_P
    -S_prewin = P_prewin + p_s_residual
    -(S_prewin + avail_aft_S ) < data_length

    P_prewin: length of time window before P arrival
    return P_prewin
    
    """
    avail_bef_P /= dt
    avail_aft_S /= dt

    P_avail_space = np.arange(
        avail_bef_P, (data_length - p_s_residual - avail_aft_S), 1
    )
    P_prewin = np.random.choice(P_avail_space)
    return P_prewin


def sac_len_complement(wf, max_length=None):
    """Complement sac data into the same length
    """
    wf_n = np.array([len(i.data) for i in wf])
    if not max_length:
        max_n = np.max(wf_n)
    else:
        max_n = max_length

    append_wf_id = np.where(wf_n != max_n)[0]
    for w in append_wf_id:
        append_npts = max_n - len(wf[w].data)
        if append_npts > 0:
            for p in range(append_npts):
                wf[w].data = np.insert(wf[w].data, -1, wf[w].data[-1])
        elif append_npts < 0:
            wf[w].data = wf[w].data[:max_n]
    return wf


def snr_pt_v2(
    tr_vertical,
    tr_horizontal,
    pt_p,
    pt_s,
    mode="std",
    snr_pre_window=5,
    snr_post_window=5,
    highpass=None,
):
    """
    Calculate snr
    tr_vertical: sac trace vertical component
    tr_horizontal: sac trace horizontal component
    pt_p: p phase utcdatetime object
    pt_s: s phase udtdatetime object
    """
    if highpass:
        tr_vertical = tr_vertical.filter("highpass", freq=highpass).taper(
            max_percentage=0.1, max_length=0.1
        )
        tr_horizontal = tr_horizontal.filter("highpass", freq=highpass).taper(
            max_percentage=0.1, max_length=0.1
        )
    tr_signal_p = tr_vertical.copy().slice(pt_p, pt_p + snr_pre_window)
    tr_signal_s = tr_horizontal.copy().slice(pt_s, pt_s + snr_pre_window)
    tr_noise_p = tr_vertical.copy().slice(pt_p - snr_pre_window, pt_p)
    tr_noise_s = tr_horizontal.copy().slice(pt_s - snr_pre_window, pt_s)

    if mode.lower() == "std":
        snr_p = np.std(tr_signal_p.data) / np.std(tr_noise_p.data)
        snr_s = np.std(tr_signal_s.data) / np.std(tr_noise_s.data)

    elif mode.lower() == "sqrt":
        snr_p = np.sqrt(np.square(tr_signal_p.data).sum()) / np.sqrt(
            np.square(tr_noise_p.data).sum()
        )
        snr_s = np.sqrt(np.square(tr_signal_s.data).sum()) / np.sqrt(
            np.square(tr_noise_s.data).sum()
        )

    return snr_p, snr_s


def stream_from_h5(dataset):
    """
    input: hdf5 dataset
    output: obspy stream

    """
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs["trace_start_time"])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs["receiver_type"] + "E"
    tr_E.stats.station = dataset.attrs["receiver_code"]
    tr_E.stats.network = dataset.attrs["network_code"]

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs["trace_start_time"])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs["receiver_type"] + "N"
    tr_N.stats.station = dataset.attrs["receiver_code"]
    tr_N.stats.network = dataset.attrs["network_code"]

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs["trace_start_time"])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs["receiver_type"] + "Z"
    tr_Z.stats.station = dataset.attrs["receiver_code"]
    tr_Z.stats.network = dataset.attrs["network_code"]

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream


def zero_pad_stream(
    slice_st, data_length, zero_pad_range, max_pad_slices=4, pad_mode="zeros"
):
    """
    Randomly pad the noise waveform with zero values on all channels

    """
    zero_pad = np.random.randint(zero_pad_range[0], zero_pad_range[1])
    max_pad_seq_num = np.random.randint(max_pad_slices) + 1
    pad_len = np.random.multinomial(
        zero_pad, np.ones(max_pad_seq_num) / max_pad_seq_num
    )

    max_v = [1.5 * np.max(slice_st[ch].data) for ch in range(3)]
    min_v = [1.5 * np.min(slice_st[ch].data) for ch in range(3)]
    for ins in range(len(pad_len)):
        max_idx = data_length - pad_len[ins]
        insert_idx = np.random.randint(max_idx)
        for ch in [0, 1, 2]:
            insert_end_idx = insert_idx + pad_len[ins]
            if insert_end_idx >= zero_pad_range[1]:
                insert_end_idx = zero_pad_range[1]
            if pad_mode == "zeros":
                slice_st[ch].data[insert_idx:insert_end_idx] = 0
            elif pad_mode == "maximum":
                slice_st[ch].data[insert_idx:insert_end_idx] = max_v[ch]
            elif pad_mode == "minimum":
                slice_st[ch].data[insert_idx:insert_end_idx] = min_v[ch]
    return slice_st


def drop_channel(slice_st, data_length, drop_chn=[0, 1]):
    for s in range(len(slice_st)):
        if s in drop_chn:
            slice_st[s].data = np.zeros(data_length)
    return slice_st
