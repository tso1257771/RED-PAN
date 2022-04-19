# RED-PAN
Real-time Earthquake Detection and Phase-picking with multi-task Attention Network

In this repository, we provide four template scripts:<br />

(1) temp_01_gen_sample_from_INSTANCE<br /> 
generate some samples using Earthquake Early Warning Augmentation (EEWA) and Marching Mosaic Waveform Augmentation (MMWA) using [INSTANCE](https://github.com/INGV/instance) dataset.<br />

(2) temp_02_model_pretrain<br />
fine-tune the trained model using few samples generated in temp_01_gen_sample_from_INSTANCE<br />

(3) temp_03_continuous_predict<br />
process continuous data using Seismogram-Tracking Medium Filter (STMF) <br />

(4) temp_04_realtime_predict<br />
triggering P arrivals using information-clipped earthquake waveform, simulating the conditions of processing real-time data.
