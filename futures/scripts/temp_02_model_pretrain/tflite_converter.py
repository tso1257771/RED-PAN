import os
import sys

sys.path.append("../REDPAN_tools")
import logging
import tensorflow as tf
from mtan_ARRU import unets

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s : %(asctime)s : %(message)s"
)

mdl_dir = "./trained_model"
mdl_hdr = "pretrain_REDPAN_60s"
data_length = 6000
frame = unets(input_size=(data_length, 3))
model = frame.build_mtan_R2unet(
    os.path.join(mdl_dir, mdl_hdr, "train.hdf5"), input_size=(data_length, 3)
)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

model_n = os.path.join(mdl_dir, mdl_hdr, mdl_hdr + ".tflite")
# Save the model.
with open(model_n, "wb") as f:
    f.write(tflite_model)
    logging.info(f"Converted: {model_n}")
