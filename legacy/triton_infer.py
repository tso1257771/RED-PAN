
## make inference through http or gRPC
# https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_infer_client.py
import os
import logging
import numpy as np
import tensorflow as tf
import tritonclient.grpc as grpcclient

# make seismic data
import numpy as np
from obspy import read

batch_size = 800

def fake_stream(batch_size: int):
    st = read()
    for s in st:
        s.data -= np.mean(s.data)
        s.data /= np.std(s.data)
    st_data = np.transpose(
        np.array([s.data for s in st])[np.newaxis, ...], [0, 2, 1])
    req_data = np.repeat(st_data, batch_size, axis=0).astype(np.float32)  
    return req_data

try:
    triton_client = grpcclient.InferenceServerClient(
        url="localhost:7001", verbose=0)
    test_output_pick = grpcclient.InferRequestedOutput("picker")
    test_output_mask = grpcclient.InferRequestedOutput("detector")
except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)


req_data = fake_stream(batch_size=batch_size)
test_input = grpcclient.InferInput("input", req_data.shape, datatype="FP32")
#test_input = grpcclient.InferInput("INPUT", req_data.shape, datatype="FP32")
test_input.set_data_from_numpy(req_data)

from time import time
t_ = []
bs = []
for i in range(100):
    print(f"Step: {i}")
    t1 = time()

    ## initialize tensor for each inference
    # _img = np.random.random([1000, 3000, 3]).astype(np.float32)
    # test_input.set_data_from_numpy(_img)
    test_input.set_data_from_numpy(req_data)
    results = triton_client.infer(model_name="onnx_RP30", 
        inputs=[test_input], 
        outputs=[test_output_pick, test_output_mask])
    pick_result = results.as_numpy('picker')
    mask_result = results.as_numpy('detector')
    t2 = time()
    print(f"Cost {t2-t1:.2f} s")
    t_.append(t2-t1)
print(f"Mean time: {np.mean(t_):.2f}")

"""

## make inference on-premise
import tensorflow as tf
import numpy as np
from time import time
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def get_func_from_saved_model(saved_model_dir):
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func, saved_model_loaded

def get_random_input(batch_size, seq_length):
    # Generate random input data
    tensor = tf.convert_to_tensor(
        np.ones((batch_size, seq_length, 3), dtype=np.float32))
    return {'input': tensor}


trt_path = '/home/rick/SSD_4T/RED-PAN/model_repository/RP30/1/model.savedmodel'
trt_func, _ = get_func_from_saved_model(trt_path)
input_tensor = get_random_input(batch_size=batch_size, seq_length=3000)

rdn_inp = [get_random_input( batch_size=np.random.randint(batch_size), 
    seq_length=3000) for i in range(10)]
## Let's run some inferences!
t_ = []
for i in range(0, 10):
   print(f"Step: {i}")
   t1 = time()
   print(rdn_inp[i]['input'].shape)
   #results = trt_func(**rdn_inp[i])
   results = trt_func(**input_tensor)
   t_.append(time()-t1)
print(f"Mean time: {np.mean(t_):.2f}")

"""