import os
import sys
sys.path.append("../")
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 12
from time import time
from glob import glob
from obspy import read, UTCDateTime
from copy import deepcopy
from scipy.signal import find_peaks
from obspy.signal.trigger import trigger_onset
from REDPAN_tools.data_utils import snr_joints
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s : %(asctime)s : %(message)s"
)
outtrg_path = "./out_trg"
if not os.path.exists(outtrg_path):
    os.makedirs(outtrg_path)

# model path
mdl_hdr = "REDPAN_60s"
model_dir = f"../pretrained_model/{mdl_hdr}"
# load tflite model
model_h5 = os.path.join(model_dir, f"{mdl_hdr}.tflite")
model = tf.lite.Interpreter(model_h5)
model.allocate_tensors()
# Get input and output tensors.
input_details = model.get_input_details()
output_details = model.get_output_details()
plt_fig = True
pred_interval_sec = 0.5
dt = 0.01
pred_npts = 6000
# only trigger P arrivals in the last 1 second of data with 100Hz sampling rate
real_time_p_trg_search = [5900, 6000]
p_snr_thre = 1

sta = "WCS2"
# define output path for figure and text
fig_path = f"./{outtrg_path}/fig_trg_20190704_{sta}_{mdl_hdr}"
os.system(f"rm -rf {fig_path}")
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
trg_out = os.path.join(outtrg_path, f"trg_20190704_{sta}.txt")
trg_f = open(trg_out, "w")
trg_f.write("network, station, triggering_utc, rt_p_utc, rt_p_pb\n")

# read continuous waveform of single station
wf = read(f"./wf_data/*.{sta}.*")
info = wf[0].stats
wf_starttime = UTCDateTime("2019-07-04T17:32:40")
wf = wf.slice(wf_starttime, wf_starttime + 240)
n_slices = int(np.ceil((len(wf[0].data) - pred_npts) / (pred_interval_sec / dt)))

init_pt = 0
P_collect = np.zeros(len(wf[0].data))
S_collect = np.zeros(len(wf[0].data))
M_collect = np.zeros(len(wf[0].data))

pred_array_P = [[] for _ in range(pred_npts)]
pred_array_S = [[] for _ in range(pred_npts)]
pred_array_mask = [[] for _ in range(pred_npts)]
pred_interval_pt = int(round(pred_interval_sec / dt))

process_t = []
trg_pool = []
trg_pool_confidence = []
trg_utc_pool = []
trg_frame_utc_pool = []
target_trg_space = -999
for i in range(n_slices):
    logging.info(f"Processing segments: {i+1}/{n_slices}")
    if init_pt + pred_npts > len(wf[0].data):
        continue

    t1 = time()
    sWF = deepcopy(wf)
    if init_pt + pred_npts > sWF[0].stats.npts:
        wf_E = sWF[0].data[init_pt:]
        wf_N = sWF[1].data[init_pt:]
        wf_Z = sWF[2].data[init_pt:]
        # insert zeros for insufficient samples
        wf_E = np.insert(wf_E, len(wf_E), np.zeros(pred_npts - len(wf_E)))
        wf_N = np.insert(wf_N, len(wf_N), np.zeros(pred_npts - len(wf_N)))
        wf_Z = np.insert(wf_Z, len(wf_Z), np.zeros(pred_npts - len(wf_Z)))
    else:
        wf_E = sWF[0].data[init_pt : init_pt + pred_npts]
        wf_N = sWF[1].data[init_pt : init_pt + pred_npts]
        wf_Z = sWF[2].data[init_pt : init_pt + pred_npts]
        z_trc = deepcopy(wf_Z)

    # buffer trace for estimating snr
    buffer_npts = 400
    if init_pt < buffer_npts:
        z_trc = deepcopy(sWF[2].data[init_pt : init_pt + pred_npts + buffer_npts])
        buffer_npts = 0
    elif (
        init_pt >= buffer_npts
        and init_pt + pred_npts + buffer_npts <= sWF[0].stats.npts
    ):
        z_trc = deepcopy(
            sWF[2].data[init_pt - buffer_npts : init_pt + pred_npts + buffer_npts]
        )
    else:
        z_trc = deepcopy(sWF[2].data[init_pt - buffer_npts : init_pt + pred_npts])

    # Z-score standardization
    for w in [wf_E, wf_N, wf_Z]:
        w -= np.mean(w)
        w /= np.std(w)
        w[np.isnan(w)] = 1e-5
        w[np.isinf(w)] = 1e-5

    model.set_tensor(
        input_details[0]["index"],
        np.stack([wf_E, wf_N, wf_Z], -1)[np.newaxis, ...].astype("float32"),
    )
    model.invoke()
    pred = model.get_tensor(output_details[0]["index"])
    mask = model.get_tensor(output_details[1]["index"])

    # split predictions into lists of lists
    pp = np.array_split(pred[0].T[0], pred_npts)
    ss = np.array_split(pred[0].T[1], pred_npts)
    mm = np.array_split(mask[0].T[0], pred_npts)

    # check current P pdf
    rt_p = pred[0].T[0]
    rt_s = pred[0].T[1]

    search_range = trigger_onset(rt_p, 0.05, 0.05, max_len=150)
    if len(search_range) > 0:
        search_range = search_range[
            np.array([(s[1] - s[0]) > 10 for s in search_range])
        ]

    p_peaks, p_peaks_info = find_peaks(rt_p, height=0.3, distance=int(1 / dt))
    p_peaks_confidence = p_peaks_info["peak_heights"]

    bool_p_peaks = []
    for _p in p_peaks:
        check_no_S_trg = np.any(
            rt_s[int(_p - 0.2 / dt) : int(int(_p + 0.2 / dt))] > 0.3
        )
        check_trgP = len(
            np.where(np.array([_p in range(s[0], s[1]) for s in search_range]))[0]
        )
        if check_trgP > 0 and check_no_S_trg == False:
            bool_p_peaks.append(True)
        else:
            bool_p_peaks.append(False)

    if len(bool_p_peaks) > 0:
        p_peaks = p_peaks[np.array(bool_p_peaks)]
        p_rt_pb = p_peaks_confidence[np.array(bool_p_peaks)]
    else:
        p_peaks = []
        p_rt_pb = []

    # if any P is detected
    if len(p_peaks) < 1:
        pass
    else:
        # stop
        current_p_pt = p_peaks + init_pt
        # estimate local SNR
        p_snrs, max_signal_amp = snr_joints(
            z_trc,
            jt_pt=p_peaks + buffer_npts,
            snr_win=1,
            dt=dt,
            mode="std",
            bandpass=True,
            freqmin=1,
            freqmax=45,
        )

        # issue alert
        for t in range(len(p_peaks)):
            # only trigger a P arrival when it is detected
            # within the specified data space under prediction window
            if not p_peaks[t] in np.arange(
                real_time_p_trg_search[0], real_time_p_trg_search[1]
            ):
                continue
            # if any activated alert has been issued with
            #  3 seconds error
            check_issued = (
                len(
                    np.where(
                        np.abs((p_peaks[t] + init_pt) - np.array(trg_pool))
                        < (real_time_p_trg_search[1] - real_time_p_trg_search[0])
                    )[0]
                )
                > 0
            )

            trg_pt = p_peaks[t] + init_pt

            if len(trg_pool) == 0:
                check_issued = False

            if np.logical_and(check_issued == False, p_snrs[t] > p_snr_thre):
                trg_pool.append(trg_pt)
                trg_utc_pool.append(wf_starttime + trg_pt * dt)
                trg_frame_utc_pool.append(
                    wf_starttime + pred_npts * dt + pred_interval_sec * (i + 1)
                )
                trg_pool_confidence.append(p_rt_pb[t])

    j = 0
    for p, s, m in zip(pp, ss, mm):
        pred_array_P[j].append(p)
        pred_array_S[j].append(s)
        pred_array_mask[j].append(m)
        j += 1

    # n*(pred_interval_sec/dt) (Since model length is an odd number 2001)
    if init_pt > 0:
        P_collect[init_pt] = np.median(pred_array_P[0][:-1])
        S_collect[init_pt] = np.median(pred_array_S[0][:-1])
        M_collect[init_pt] = np.median(pred_array_mask[0][:-1])
        # print(f"take median from {len(pred_array_mask[0][:-1])} values")
    else:
        P_collect[init_pt] = np.median(pred_array_P[0])
        S_collect[init_pt] = np.median(pred_array_S[0])
        M_collect[init_pt] = np.median(pred_array_mask[0])
        # print(f"take median from {len(pred_array_mask[0])} values")

    # others
    P_collect[init_pt + 1 : init_pt + pred_interval_pt] = np.hstack(
        np.median(pred_array_P[1:pred_interval_pt], axis=1)
    )
    S_collect[init_pt + 1 : init_pt + pred_interval_pt] = np.hstack(
        np.median(pred_array_S[1:pred_interval_pt], axis=1)
    )
    M_collect[init_pt + 1 : init_pt + pred_interval_pt] = np.hstack(
        np.median(pred_array_mask[1:pred_interval_pt], axis=1)
    )

    # dequeue taken values and enqueue empty lists for unseen data
    pred_array_P[: pred_npts - pred_interval_pt] = pred_array_P[pred_interval_pt:]
    pred_array_P[pred_npts - pred_interval_pt :] = [[] for _ in range(pred_interval_pt)]
    pred_array_S[: pred_npts - pred_interval_pt] = pred_array_S[pred_interval_pt:]
    pred_array_S[pred_npts - pred_interval_pt :] = [[] for _ in range(pred_interval_pt)]
    pred_array_mask[: pred_npts - pred_interval_pt] = pred_array_mask[pred_interval_pt:]
    pred_array_mask[pred_npts - pred_interval_pt :] = [
        [] for _ in range(pred_interval_pt)
    ]

    try:
        t2 = time()
        process_t.append(t2 - t1)

        if plt_fig:  # and i == n_slices-1:
            # fig, ax = plt.subplots(5, 1, sharex=True, figsize=(8, 8))
            fig = plt.figure(figsize=(8, 6))
            ax_grd = gridspec.GridSpec(5, 1, figure=fig, hspace=0)
            ax = [fig.add_subplot(ax_grd[j, 0]) for j in range(5)]
            _P, _S, _M = (
                np.zeros(wf[0].stats.npts),
                np.zeros(wf[0].stats.npts),
                np.zeros(wf[0].stats.npts),
            )
            _P[init_pt : init_pt + pred_npts] = pred[0].T[0]
            _S[init_pt : init_pt + pred_npts] = pred[0].T[1]
            _M[init_pt : init_pt + pred_npts] = mask[0].T[0]
            dt = 0.01
            _t = np.arange(wf[0].stats.npts) * dt

            # pdfs
            ax[0].plot(_t, _P, linewidth=1, label="P", color="k")
            ax[0].plot(_t, _S, linewidth=1, label="S", color="r")
            ax[0].plot(_t, _M, linewidth=1, label="mask", color="g")
            # ax[0].axvline(p_npts, linewidth=1, label='targetP', color='k')
            ax[0].legend()
            ax[0].set_title(f"Prediction interval: {pred_interval_sec} s", fontsize=12)
            ax[0].set_ylabel(f"Real time\nPDFs")
            ax[0].set_ylim(-0.1, 1.15)
            ax[0].set_xlim(0, wf[0].stats.npts * dt)
            ax[0].axvline(
                (init_pt + pred_npts) * dt,
                color="red",
                label="current time",
                linewidth=1,
            )
            # ax[0].axvline(init_pt, color='r', linestyle=':',
            #    label='prediction frame start time')
            ax[0].axvspan(
                init_pt * dt, (init_pt + pred_npts) * dt, color="r", alpha=0.1
            )

            if len(trg_pool) > 0:
                for T in range(len(trg_pool)):
                    ax[0].axvline(
                        trg_pool[T] * dt, linestyle=":", color="k", linewidth=1
                    )
                    ax[0].text(trg_pool[T] * dt, 1, f"trg")
            # processed pdfs
            ax[1].plot(_t, P_collect[:], linewidth=1, color="k")
            ax[1].plot(_t, S_collect[:], linewidth=1, color="r")
            ax[1].plot(_t, M_collect[:], linewidth=1, color="g")
            ax[1].axvspan(
                init_pt * dt,
                (init_pt + pred_interval_pt) * dt,
                color="b",
                alpha=0.2,
                label="renewed PDF",
                linewidth=1,
            )
            ax[1].legend()
            ax[1].set_ylabel(f"Renewed\nPDFs")
            ax[1].set_ylim(-0.1, 1.15)
            ax[1].set_xlim(0, wf[0].stats.npts * dt)
            # waveform
            ax[2].plot(_t, wf[0].data, linewidth=1)
            ax[3].plot(_t, wf[1].data, linewidth=1)
            ax[4].plot(_t, wf[2].data, linewidth=1)
            for r in range(2, 5):
                # current prediction frame
                ax[r].axvline(
                    (init_pt + pred_npts) * dt,
                    color="r",
                    label="current time",
                    linewidth=1,
                )
                ax[r].axvspan(
                    init_pt * dt,
                    (init_pt + pred_npts) * dt,
                    color="r",
                    alpha=0.1,
                    label="prediction frame",
                )
                ax[r].set_xlim(0, wf[0].stats.npts * dt)
            for r2 in range(4):
                ax[r2].set_xticks([])
            ax[2].legend(ncol=3)
            ax[2].set_yticks([])
            ax[3].set_yticks([])
            ax[4].set_yticks([])
            ax[2].set_ylabel("E")
            ax[3].set_ylabel("N")
            ax[4].set_ylabel("Z")
            ax[4].set_xlabel("Time (s)")
            plt.tight_layout()
            plt.savefig(f"{fig_path}/{i:07}.png", dpi=150)
            # plt.show()
            plt.close()

        init_pt += pred_interval_pt
    except:
        plt.close()
        stop
        # continue

# write every real-time triggered P arrivals
if len(trg_pool) != 0:
    for ts in range(len(trg_pool)):
        trg_p_utc = trg_utc_pool[ts]
        trg_frame_utc = trg_frame_utc_pool[ts]
        trg_info = (
            f"{info.network}, {info.station}, "
            + f"{trg_frame_utc}, "
            + f"{trg_p_utc}, {trg_pool_confidence[ts]:.2f}\n"
        )
        trg_f.write(trg_info)
trg_f.close()

print(f"logged: {trg_out}")
print(f"Mean process time per inference and buffering: {np.mean(process_t)}")
