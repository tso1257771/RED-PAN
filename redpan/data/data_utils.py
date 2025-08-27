import gc
import numpy as np
import multiprocessing
from scipy.signal.windows import tukey
from scipy.signal import find_peaks
from obspy import read, Stream
from obspy.signal import filter
from obspy.signal.filter import envelope
from obspy.signal.trigger import trigger_onset
from copy import deepcopy

def snr_joints(
    z_trc, jt_pt, snr_win, dt=0.01, mode="std", bandpass=False, freqmin=1, freqmax=45
):
    """Calculate SNR of every point on the vertical component trace
    """
    joint_snrs = []
    max_signal_amp = []
    for jt in range(len(jt_pt)):
        # store the maximum amplitude value of the signal
        # for further estimation, e.g. Pd
        max_signal_amp.append(
            np.max(np.abs(z_trc[jt_pt[jt] : int(jt_pt[jt] + snr_win / dt)]))
        )

        if bandpass:
            bp_trc = tukey(len(z_trc), alpha=0.1) * (
                filter.bandpass(deepcopy(z_trc), freqmin=1, freqmax=45, df=1 / dt)
            )
        else:
            bp_trc = z_trc

        sig = bp_trc[jt_pt[jt] : int(jt_pt[jt] + snr_win / dt)]
        nz = bp_trc[int(jt_pt[jt] - snr_win / dt) : jt_pt[jt]]
        if mode.lower() == "std":
            n_std = np.std(nz)
            if n_std == 0:
                n_std = 1
            joint_snrs.append(np.std(sig) / n_std)
        elif mode.lower() == "sqrt":
            n_sqrt = np.sqrt(np.sum(np.abs(nz) ** 2))
            if n_sqrt == 0:
                n_sqrt = 1
            joint_snrs.append(np.sqrt(np.sum(np.abs(sig) ** 2)) / n_sqrt)
        else:
            raise Exception("mode must be 'std' or 'sqrt'")

    return joint_snrs, max_signal_amp


def mosaic_Psnr_npts(
    z_trc, gt_label_P, P_snr_win=3, dt=0.01, data_length=2001, hpfreq=2, mode="sqrt"
):
    if hpfreq:
        _hpZtrc = tukey(data_length, alpha=0.1) * (
            filter.highpass(z_trc, freq=hpfreq, df=1 / dt)
        )
    else:
        _hpZtrc = z_trc

    _P_snr_win = int(P_snr_win / 0.01)
    Psnrs = []
    for sufP in range(len(gt_label_P)):
        if gt_label_P[sufP] < _P_snr_win:
            snr_win = gt_label_P[sufP] - 1
        elif gt_label_P[sufP] + _P_snr_win > data_length:
            snr_win = data_length - gt_label_P[sufP] - 1
        else:
            snr_win = _P_snr_win

        sig = _hpZtrc[gt_label_P[sufP] : gt_label_P[sufP] + snr_win]
        nz = _hpZtrc[gt_label_P[sufP] - snr_win : gt_label_P[sufP]]
        if mode.lower() == "std":
            tr_noise = np.std(nz)
            tr_pt = np.std(sig)
        elif mode.lower() == "sqrt":
            tr_noise = np.sqrt(np.square(nz).sum())
            tr_pt = np.sqrt(np.square(sig).sum())
        Psnrs.append(tr_pt / tr_noise)
    return Psnrs


def MWA_suffix_Psnr(
    trc_mosaic, tp_to_ori_pt, snr_win=0.5, dt=0.01, hpfreq=2, mode="std"
):
    """Calculate SNR of suffix P on merged waveforms
    """
    suffix_Psnrs = []
    data_npts = len(trc_mosaic[0])
    for sufP in range(len(tp_to_ori_pt[1:])):
        _hpZtrc = tukey(data_npts, alpha=0.1) * (
            filter.highpass(trc_mosaic[2], freq=hpfreq, df=1 / dt)
        )
        sufP_pt = tp_to_ori_pt[sufP + 1]
        if sufP_pt >= data_npts:
            suffix_Psnrs.append(999)
        else:
            sig = _hpZtrc[sufP_pt : int(sufP_pt + snr_win / dt)]
            nz = _hpZtrc[int(sufP_pt - snr_win / dt) : sufP_pt]
            if mode.lower() == "std":
                tr_noise = np.std(nz)
                tr_pt = np.std(sig)
            elif mode.lower() == "sqrt":
                tr_noise = np.sqrt(nz.sum())
                tr_pt = np.sqrt(sig.sum())
            suffix_Psnrs.append(tr_pt / tr_noise)
    return np.array(suffix_Psnrs)


def MWA_joint_Zsnr(trc_mosaic, cumsum_npts, snr_win=1, dt=0.01, hpfreq=2, mode="std"):
    """Calculate SNR of waveform joint on vertical component
    """
    joint_snrs = []
    data_npts = len(trc_mosaic[0])
    for jt in range(len(cumsum_npts[:-1])):
        _hpZtrc = tukey(data_npts, alpha=0.1) * (
            filter.highpass(trc_mosaic[2], freq=hpfreq, df=1 / dt)
        )
        jt_pt = cumsum_npts[jt]
        sig = _hpZtrc[jt_pt : int(jt_pt + snr_win / dt)]
        nz = _hpZtrc[int(jt_pt - snr_win / dt) : jt_pt]
        if mode.lower() == "std":
            tr_noise = np.std(nz)
            tr_pt = np.std(sig)
        elif mode.lower() == "sqrt":
            tr_noise = np.sqrt(nz.sum())
            tr_pt = np.sqrt(sig.sum())
        joint_snrs.append(tr_pt / tr_noise)
    return np.array(joint_snrs)


def MWA_joint_ENZsnr(trc_mosaic, joint_pt, snr_win=1, dt=0.01, hpfreq=2, mode="std"):
    """Calculate SNR of waveform joint on vertical component
    """
    chn_joint_snrs = []
    for ms in trc_mosaic:
        joint_snrs = []
        data_npts = len(trc_mosaic[0])
        for jt in range(len(joint_pt)):
            _hptrc = tukey(data_npts, alpha=0.1) * (
                filter.highpass(ms, freq=hpfreq, df=1 / dt)
            )
            jt_pt = joint_pt[jt]
            sig = _hptrc[jt_pt : int(jt_pt + snr_win / dt)]
            nz = _hptrc[int(jt_pt - snr_win / dt) : jt_pt]
            if mode.lower() == "std":
                tr_noise = np.std(nz)
                tr_pt = np.std(sig)
            elif mode.lower() == "sqrt":
                tr_noise = np.sqrt(nz.sum())
                tr_pt = np.sqrt(sig.sum())
            joint_snrs.append(tr_pt / tr_noise)
        chn_joint_snrs.append(joint_snrs)
    return np.array(chn_joint_snrs)

if __name__ == "__main__":
    pass
