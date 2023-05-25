import os
import sys
sys.path.append("../")
import tensorflow as tf

from REDPAN_tools.mtan_ARRU import unets
from REDPAN_tools.data_utils import PhasePicker

def redpan_picker(
        model_path, 
        pred_interval_sec=10, 
        dt=0.01, 
        postprocess_config=None
        ):
    """
    :param model_h5_path:
    :param postprocess_config:
    :return:
    """
    # load model
    try:
        mdl_hdr = os.path.basename(model_path)
        if mdl_hdr[-3:] == '30s':
            pred_npts = 3000
        elif mdl_hdr[-3:] == '60s':
            pred_npts = 6000
        # load model and weights
        frame = unets()
        model = frame.build_mtan_R2unet(
            os.path.join(model_path, 'train.hdf5'), 
            input_size=(pred_npts, 3)
        )

        picker = PhasePicker(
            model=model, 
            pred_npts=pred_npts,
            pred_interval_sec=pred_interval_sec,
            dt=dt, 
            postprocess_config=postprocess_config
        )

        print(f"Loaded PhasePicker: {mdl_hdr}")
        print(f"Model path: {os.path.abspath(model_path)}")
    except Exception as e:
        print(f"Failed to load PhasePicker: {mdl_hdr}")
        print(f"Maybe the ${model_path} is wrong")
        print(e)
        return None, None
    return picker, pred_npts
