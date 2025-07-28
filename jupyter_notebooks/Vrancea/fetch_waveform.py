#!/usr/bin/env python
import logging
import os
import obspy
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
logging.basicConfig(level=logging.INFO,
    format='%(levelname)s : %(asctime)s : %(message)s')

client = Client("NIEP")
odir = './outputs/SAC'
##################### Config #####################
pi = 3.1415926
degree2km = pi * 6371 / 180

center = (26.5, 45.5) 
region = [25, 28, 44.5, 46.5]
horizontal_degree = 3 
vertical_degree = 2

# Romania Vrancea County Mw5.3 earthquake 2016-09-23T23:11 (2016.267.00)
starttime = UTCDateTime("2016-09-23T00:00:00") 
endtime = UTCDateTime("2016-09-25T00:00:00")

## seismic stations
network_list = ["AM", "BS", "GE", "MD","OX", "RO", "UD", "Y8"]
channel_list = "HH*,HN*,EH*"

## data center
config = {}
config["center"] = center
config["xlim_degree"] = [center[0] - horizontal_degree / 2, center[0] + horizontal_degree / 2]
config["ylim_degree"] = [center[1] - vertical_degree / 2, center[1] + vertical_degree / 2]
config["degree2km"] = degree2km
config["starttime"] = starttime.datetime
config["endtime"] = endtime.datetime
config["networks"] = network_list
config["channels"] = channel_list
config["client"] = client
####################################################
stations = client.get_stations(
    network=",".join(config["networks"]),
    #station=",".join(stations_select["station"]),
    starttime=config["starttime"],
    endtime=config["endtime"],
    minlongitude=config["xlim_degree"][0],
    maxlongitude=config["xlim_degree"][1],
    minlatitude=config["ylim_degree"][0],
    maxlatitude=config["ylim_degree"][1],
    channel=config["channels"],
    level="response"
)

def stations_info(stations, starttime, endtime):
    stations = client.get_stations(
        network=",".join(config["networks"]),
        #station=",".join(stations_select["station"]),
        starttime=starttime,
        endtime=endtime,
        minlongitude=config["xlim_degree"][0],
        maxlongitude=config["xlim_degree"][1],
        minlatitude=config["ylim_degree"][0],
        maxlatitude=config["ylim_degree"][1],
        channel=config["channels"],
        level="response",
    ) 

    station_locs = defaultdict(dict)
    station_resp = defaultdict(dict)
    station_pz = defaultdict(dict)
    for network in stations:
        for station in network:
            for chn in station:
                sid = f"{network.code}.{station.code}.{chn.location_code}.{chn.code[:-1]}"
                station_resp[
                    f"{network.code}.{station.code}.{chn.location_code}.{chn.code}"
                ] = chn.response.instrument_sensitivity.value
                if sid in station_locs:
                    station_locs[sid]["component"] += f",{chn.code[-1]}"
                    station_locs[sid]["response"] += \
                        f",{chn.response.instrument_sensitivity.value:.2f}"
                else:
                    component = f"{chn.code[-1]}"
                    response = f"{chn.response.instrument_sensitivity.value:.2f}"
                    dtype = chn.response.instrument_sensitivity.input_units.lower()
                    tmp_dict = {}
                    tmp_dict["longitude"], tmp_dict["latitude"], tmp_dict["elevation(m)"] = (
                        chn.longitude,
                        chn.latitude,
                        chn.elevation,
                    )
                    tmp_dict["component"], tmp_dict["response"], tmp_dict["unit"] = \
                        component, response, dtype
                    station_locs[sid] = tmp_dict
                    station_pz[sid] = chn.response.get_sacpz()

    station_locs = pd.DataFrame.from_dict(station_locs, orient='index')
    return station_locs, station_resp, station_pz

multiprocess_n = 4
jday_start = starttime.julday
jday_end = endtime.julday
jidx_id = np.array([f"2016,{i}" for i in range(jday_start, jday_end+1)])

# loop over hour
for d in range(len(jidx_id)):
    logging.info(f"{d+1}/{len(jidx_id)}: {jidx_id[d]}")
    yr, jday = jidx_id[d].split(',')

    outdir = os.path.join(odir, f"{int(yr):04}.{int(jday):03}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    buffer_sec = 60
    stt = UTCDateTime(year=int(yr), julday=int(jday)) - buffer_sec
    ent = stt + buffer_sec*2 + 3600

    station_locs = stations_info(
        stations, 
        starttime=stt, 
        endtime=ent
    )

    station_locs = station_locs.drop_duplicates()
    station_id_all = np.array(station_locs.index)

    # split the station list for multiprocessing
    process_sta_n = len(station_id_all)
    process_idx = np.array_split(station_id_all, multiprocess_n)
    mp_args = [(station_locs, station_ids, stt, ent, outdir) 
               for station_ids in process_idx]

    def process_station(station_locs, station_ids, stt, ent, outdir):
        """
        Process each station and download the waveform data.
        """
        for i in range(len(station_ids)):
            comp = np.unique(station_locs.component[station_ids[i]].split(','))
            net, sta, loc, chn = station_ids[i].split('.')
            # loop over channel
            for j in range(len(comp)):
                out_file = os.path.join(outdir, f"{net}.{sta}.{loc}.{chn+comp[j]}.sac")
                if os.path.exists(out_file):
                    print(f"File exists: {out_file}, skipping...")
                    continue
                try:
                    try:
                        st = client.get_waveforms(net, sta, loc, chn+comp[j], 
                            starttime=stt, endtime=ent)
                        print(f"Available: {station_ids[i]+comp[j]} {stt} - {ent}")
                    except: 
                        st = client.get_waveforms(net, sta, loc, chn+comp[j], 
                            starttime=stt, endtime=ent)
                        print(f"Available: {station_ids[i]+comp[j]} {stt} - {ent}")
                except:
                    st = None
                    print(f"Not available: {station_ids[i]+comp[j]} {stt} - {ent}")
                    continue

                if len(st) > 1:
                    st = st.merge(method=0, fill_value='interpolate')

                # check length
                if not np.logical_and(
                    np.abs(stt - st[0].stats.starttime) < 180,
                    np.abs(ent - st[0].stats.endtime) < 180
                ):
                    print(f"Trace length error: {station_ids[i]+comp[j]} {stt} - {ent}")
                    continue 

                for tr in st:
                    if tr.stats.sampling_rate != 100:
                        tr = tr.interpolate(100, method="linear")                    
                    if isinstance(tr.data, np.ma.masked_array):
                        tr.data = tr.data.filled()

                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                st.write(out_file, format='SAC')

    # Multiprocessing pool
    with mp.Pool(processes=multiprocess_n) as pool:
        # Use starmap to distribute tasks among processes
        pool.starmap(process_station, mp_args)
        pool.close()
        pool.join()
