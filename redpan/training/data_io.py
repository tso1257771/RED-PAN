import os
import sys
import logging
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s : %(asctime)s : %(message)s"
)


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_TFRecord_detect(trc_3C, label_psn, mask, idx, outfile):
    """
    1. Create feature dictionary to be ready for setting up
        tf.train.Example object
        tf.train.Example can only accept 1-d data
    2. Create example protocol using tf.train.Example
    3. Write TFRecord object
    """
    feature = {
        "trc_data": _float_feature(value=trc_3C.flatten()),
        "label_data": _float_feature(value=label_psn.flatten()),
        "mask": _float_feature(value=mask.flatten()),
        "idx": _bytes_feature(value=idx.encode("utf-8")),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    out = tf.io.TFRecordWriter(outfile)
    out.write(example_proto.SerializeToString())


def _parse_function_detect(
    record,
    data_length=6000,
    trc_chn=3,
    label_chn=3,
    mask_chn=2,
    batch_size=10,
    mode="custom_train",
    drop_chn=False,
):
    flatten_size_trc = data_length * trc_chn
    flatten_size_label = data_length * label_chn
    flatten_size_mask = data_length * mask_chn
    feature = {
        "trc_data": tf.io.FixedLenFeature([flatten_size_trc], tf.float32),
        "label_data": tf.io.FixedLenFeature([flatten_size_label], tf.float32),
        "mask": tf.io.FixedLenFeature([flatten_size_mask], tf.float32),
        "idx": tf.io.FixedLenFeature([], tf.string),
    }

    record = tf.io.parse_example(record, feature)
    record["trc_data"] = tf.reshape(
        record["trc_data"], (batch_size, data_length, trc_chn)
    )
    record["label_data"] = tf.reshape(
        record["label_data"], (batch_size, data_length, label_chn)
    )
    record["mask"] = tf.reshape(record["mask"], (batch_size, data_length, mask_chn))

    if drop_chn:
        if np.random.random() < 0.5:
            d_chn = np.random.randint(3)
        else:
            d_chn = np.array([0, 1])
        for i in range(int(batch_size * 0.2)):
            for d in d_chn:
                record["trc_data"][i].T[d, :] = tf.zeros(data_length)

    if mode != "custom_train":
        return record["trc_data"], (record["label_data"], record["mask"])
    else:
        return record["trc_data"], record["label_data"], record["mask"], record["idx"]

def _parse_function_detect_1chn(
    record,
    data_length=6000,
    trc_chn=3,
    label_chn=3,
    mask_chn=2,
    batch_size=10,
    mode="custom_train",
    drop_chn=False,
):
    flatten_size_trc = data_length * trc_chn
    flatten_size_label = data_length * label_chn
    flatten_size_mask = data_length * mask_chn
    feature = {
        "trc_data": tf.io.FixedLenFeature([flatten_size_trc], tf.float32),
        "label_data": tf.io.FixedLenFeature([flatten_size_label], tf.float32),
        "mask": tf.io.FixedLenFeature([flatten_size_mask], tf.float32),
        "idx": tf.io.FixedLenFeature([], tf.string),
    }

    record = tf.io.parse_example(record, feature)
    record["trc_data"] = tf.reshape(
        record["trc_data"], (batch_size, data_length, trc_chn)
    )
    record["label_data"] = tf.reshape(
        record["label_data"], (batch_size, data_length, label_chn)
    )
    record["mask"] = tf.reshape(record["mask"], (batch_size, data_length, mask_chn))

    if drop_chn:
        if np.random.random() < 0.5:
            d_chn = np.random.randint(3)
        else:
            d_chn = np.array([0, 1])
        for i in range(int(batch_size * 0.2)):
            for d in d_chn:
                record["trc_data"][i].T[d, :] = tf.zeros(data_length)

    if mode != "custom_train":
        return record["trc_data"][:, :, 2][..., np.newaxis], \
            record["label_data"], record["mask"],
    else:
        return record["trc_data"][:, :, 2][..., np.newaxis], \
            record["label_data"], \
            record["mask"], record["idx"]

def _yield_batch_detect(
    parsed_dataset, batch_size, data_length, trc_channel, label_channel, mask_chn
):
    parsed_iterator = parsed_dataset.as_numpy_iterator()
    for ds in parsed_iterator:
        trc = tf.reshape(ds["trc_data"], (batch_size, data_length, trc_channel))
        label = tf.reshape(ds["label_data"], (batch_size, data_length, label_channel))
        idx = ds["idx"]
        yield trc, label, idx


def tfrecord_dataset_detect(
    file_list,
    repeat=-1,
    batch_size=None,
    trc_chn=3,
    label_chn=3,
    mask_chn=2,
    data_length=6000,
    shuffle_buffer_size=300,
    mode="custom_train",
    drop_chn=False,
):
    if batch_size == None:
        raise ValueError("Must specify value of `batch_size`")
    else:
        dataset = tf.data.TFRecordDataset(file_list, num_parallel_reads=AUTOTUNE)
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True
        )
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        parsed_dataset = dataset.map(
            lambda x: _parse_function_detect(
                x,
                data_length=data_length,
                trc_chn=trc_chn,
                label_chn=label_chn,
                mask_chn=2,
                batch_size=batch_size,
                mode=mode,
            ),
            num_parallel_calls=AUTOTUNE,
        )
        return parsed_dataset

def tfrecord_dataset_detect_1chn(
    file_list,
    repeat=-1,
    batch_size=None,
    trc_chn=3,
    label_chn=3,
    mask_chn=2,
    data_length=6000,
    shuffle_buffer_size=300,
    mode="custom_train",
    drop_chn=False,
):
    if batch_size == None:
        raise ValueError("Must specify value of `batch_size`")
    else:
        dataset = tf.data.TFRecordDataset(file_list, num_parallel_reads=AUTOTUNE)
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True
        )
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        parsed_dataset = dataset.map(
            lambda x: _parse_function_detect_1chn(
                x,
                data_length=data_length,
                trc_chn=trc_chn,
                label_chn=label_chn,
                mask_chn=2,
                batch_size=batch_size,
                mode=mode,
            ),
            num_parallel_calls=AUTOTUNE,
        )
        return parsed_dataset

if __name__ == "__main__":
    pass
