import os
import numpy as np


def save_model(model, outdir, train_loss_epoch, val_loss_epoch):
    # define output documents
    out_json = os.path.join(outdir, "train.json")
    out_weights = os.path.join(outdir, "train.weights")
    out_model = os.path.join(outdir, "train.hdf5")
    out_loss = os.path.join(outdir, "train_loss.npy")
    out_val_loss = os.path.join(outdir, "val_loss.npy")

    model_json = model.to_json()
    modelout_json = out_json
    modelout_weights = out_weights
    with open(modelout_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save(out_model)
    model.save_weights(modelout_weights)
    message = f"Saved model to : {out_model}"

    np.save(out_loss, np.array(train_loss_epoch))
    np.save(out_val_loss, np.array(val_loss_epoch))

    return message
