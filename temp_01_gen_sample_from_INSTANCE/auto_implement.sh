### This template demonstrates how to generate some sample data from INSTANCE dataset(https://github.com/INGV/instance)

## 1. Download the sample dataset of INSTANCE

mkdir INSTANCE

cd INSTANCE

curl http://repo.pi.ingv.it/instance/Instance_sample_dataset.tar.bz2 | tar xj

cd ../

## 2. Make prediction on some data of INSTANCE using pretrained RED-PAN(60s) model.
#   We wish to generate semi-synthetic data from waveform containing only one pair of P and S arrival.
#   We will produce SAC files with t1 and t2 labeled with RED-PAN(60s) picked P and S arrival.

python P00_slice_INSTANCE.py

## 3. Generate some samples and figures of earthquake waveform formatted in tfrecord.

python P01_TFRecord_SingleEQ.py

## 4. Generate some samples and figures of zero-padded earthquake waveform formatted in tfrecord. 

python P02_TFRecord_SingleEQ_zeropad.py

## 5. Generate some samples and figures of Marching Mosaic Waveform Augmentation (MMWA) formatted in tfrecord.

python P03_TFRecord_MMWA.py

## 6. Generate some samples and figures of Earthquake Early Warning Augmentation (EEWA) formatted in tfrecord.

python P04_TFRecors_EEWA.py


