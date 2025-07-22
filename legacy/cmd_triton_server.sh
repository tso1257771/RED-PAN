## (optional) convert tensorflow savedmodel to onnx format
cd /home/rick/SSD_4T/RED-PAN/model_repository
python -m tf2onnx.convert --saved-model ../pretrained_model/SavedModel_RP30 --output RP30.onnx

# download image
sudo docker pull nvcr.io/nvidia/tritonserver:21.06-py3

# start triton server
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/rick/SSD_4T/RED-PAN/model_repository:/models nvcr.io/nvidia/tritonserver:21.06-py3 tritonserver --model-repository=/models --backend-config=tensorflow,version=2


## Run the Triton Inference Server container
docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:21.06-py3


############# /// Example : Setup up server, client, and run inference /// #############
# // Step 1. Triton server startup. Open a terminal and command
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/rick/SSD_4T/RED-PAN/model_repository:/models nvcr.io/nvidia/tritonserver:21.06-py3 tritonserver --model-repository=/models --backend-config=tensorflow,version=2

# // Step 2. Enter the server. Open another terminal and access the server container, and then clone en example
$ export CONTAINER_ID=86025d9b40e6 # 範例
$ docker exec -it $CONTAINER_ID bash
# ----------------------------------
# 進入到容器，當前目錄在 /opt/tritonserver#

# 下載 python-based model 範例
$ git clone https://github.com/triton-inference-server/python_backend -b r21.06
$ cd python_backend
# 複製範例程式到 /models/add_sub 底下
$ mkdir -p /models/add_sub/1/
$ cp examples/add_sub/model.py     /models/add_sub/1/model.py
$ cp examples/add_sub/config.pbtxt /models/add_sub/config.pbtxt

# // Step 3. Enter the client container

# open another docker enviroment as client (example)
docker run --name triton_client --rm -it --net host nvcr.io/nvidia/tritonserver:21.06-py3-sdk /bin/bash

# 下載 python-based 的 client
$ git clone https://github.com/triton-inference-server/python_backend -b r21.06

# 執行範例程式
$ python3 python_backend/examples/add_sub/client.py
############# /// End of example /// #####################################################

