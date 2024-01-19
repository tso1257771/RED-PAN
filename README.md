# RED-PAN
This is the official implementation of **Real-time Earthquake Detection and Phase-picking with multi-task Attention Network**<br />


https://user-images.githubusercontent.com/30610646/166941015-921d6ba1-f77e-4413-a532-e3e5af6d658f.mp4

## Summary

* [Installation](#installation)
* [Project Architecture](#project-architecture)
* [Run template codes](#run-template-codes)

### Installation
To run this repository, we suggest to install packages with Anaconda.

Clone this repository:

```bash
git clone https://github.com/tso1257771/RED-PAN.git
cd RED-PAN
```

Create a new environment via conda & pip

```bash
conda update conda
conda create --name redpan python=3.9
conda activate redpan
pip install --upgrade pip
 pip install -r requirements

```



### Project Architecture

```bash
.
├── temp_01_gen_sample_from_INSTANCE      ### *Temp01. Generate template samples with different data augmentation stratigies*
│   ├── auto_implement.sh                 # automatically implement program P00~P04*.py to generate samples from INSTANCE
│   ├── P00_slice_INSTANCE.py             # make SAC files from INSTANCE<br />
│   ├── P01_TFRecord_SingleEQ.py          # generate tfrecord files of single-earthquake waveform
│   ├── P02_TFRecord_SingleEQ_zeropad.py  # generate tfrecord files of zero-padded single-earthquake waveform
│   ├── P03_TFRecord_MMWA.py              # generate Marching Mosaic Waveform Augmentation samples
│   └── P04_TFRecors_EEWA.py              # generate Earthquake Early Warning Augmentation samples
├── temp_02_model_pretrain                ### *Temp02. Fine-tune the REDPAN(60s) with samples obtained with temp_01 outputs.* 
│   ├── P01_train_REDPAN_dwa.py           # a template code of multi-task model training using dynamic weight average (dwa) strategies
│   └── tflite_converter.py               # convert the trained model to tflite format (optional)
├── temp_03_continuous_predict            ### *Temp03. Make prediction on continuous data using Seismogram-Tracking Median Filter (STMF) strategy*
│   ├── P01_continuous_pred.py            # load a 1-hour-long sac file and make predictions
│   ├── P02_pred_result_csv.py            # write the earthquake waveform detection and picking results from the REDPAN output funtions
└── temp_04_realtime_predict              ### *Temp04. Do real-time detection simulations*
    ├── P01_rt_pred_tflite.py             # do real-time detection using tflite model and plot the result at every time step of sliding prediction
    ├── P02_make_avi.py                   # make the animation using ffmpeg package (optional)
.
```

### Run template codes
In this repository, we provide four template scripts:<br />

**(1) temp_01_gen_sample_from_INSTANCE**<br /> 
generate some samples using **Earthquake Early Warning Augmentation (EEWA)** and **Marching Mosaic Waveform Augmentation (MMWA)** using [INSTANCE](https://github.com/INGV/instance) dataset.<br />
```bash
cd temp_01_gen_sample_from_INSTANCE
bash auto_implement.sh
```

**(2) temp_02_model_pretrain**<br />
fine-tune the trained model using few samples generated in ```temp_01_gen_sample_from_INSTANCE```<br />
```bash
cd temp_02_model_pretrain
python P01_train_REDPAN_dwa.py
```

**(3) temp_03_continuous_predict**<br />
process continuous data using **Seismogram-Tracking Medium Filter (STMF)** <br />
```bash
cd temp_03_continuous_predict
python P01_continuous_pred.py
```


**(4) temp_04_realtime_predict**<br />
triggering P arrivals using information-clipped earthquake waveform, simulating the conditions of real-time data processing.

```bash
cd temp_04_realtime_predict
python P01_rt_pred_tflite.py
```

