# RED-PAN
This is the official implementation of **Real-time Earthquake Detection and Phase-picking with multi-task Attention Network**<br />

# Project updating (2025/07/22)
I am rewriting the whole RED-PAN API and application algorithms, will release soon.

https://user-images.githubusercontent.com/30610646/166941015-921d6ba1-f77e-4413-a532-e3e5af6d658f.mp4

## Summary

* [Installation](#installation)
* [Project Architecture](#project-architecture)

### Installation
To run this repository, we suggest to install packages with Anaconda.

Clone this repository:

```bash
git clone https://github.com/tso1257771/RED-PAN.git
cd RED-PAN
```

Create a new environment via pip (suggested)

```bash
conda create --name REDPAN python==3.7.3 
conda activate REDPAN
pip install -r requirements.txt
```
or via environment.yml 

```bash
conda env create --file environment.yml
conda activate REDPAN
```
