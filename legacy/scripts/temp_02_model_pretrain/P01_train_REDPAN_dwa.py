import os
import sys
sys.path.append('../REDPAN_tools')
import logging
import tensorflow as tf
import numpy as np
from glob import glob
from tensorflow_addons.optimizers import RectifiedAdam
from mtan_ARRU import unets
from model_build_utils.save import save_model
from model_build_utils.multitask_opt_model import distributed_train_step_dwa
from model_build_utils.multitask_opt_model import  distributed_val_step_dwa
from data_io import tfrecord_dataset_detect
AUTOTUNE = tf.data.experimental.AUTOTUNE
logging.basicConfig(level=logging.INFO,
    format='%(levelname)s : %(asctime)s : %(message)s')
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
use_devices = ["/gpu:0", "/gpu:1"]

############################## basic information
mdl = 'R2unet' # R2unet, attR2unet
pretrain = '../pretrained_model/REDPAN_60s'
data_length = 6000
frame = unets(input_size=(data_length, 3))
outdir = os.path.join('./trained_model', f'pretrain_REDPAN_60s')
if not os.path.exists(outdir):
    os.makedirs(outdir)

glob_list = np.random.permutation(glob(
    '../temp_01_gen_sample_from_INSTANCE/tfrecord/*/*.tfrecord'))

train_list = glob_list[:int(len(glob_list)*0.8)]
val_list = glob_list[int(len(glob_list)*0.8):]

# model training parameters
train_epoch = 10
not_improve_patience = 5
save_model_per_epoch = False
# multi-devices strategy
batch_size_per_replica = 5
strategy = tf.distribute.MirroredStrategy(devices=use_devices)
n_device = strategy.num_replicas_in_sync
global_batch_size = batch_size_per_replica * n_device
train_steps = int(np.floor(len(train_list)/\
    (batch_size_per_replica*n_device)*train_epoch))
#dwa parameters
T = 2
avg_cost = np.zeros([train_epoch, 2], dtype=np.float32)
lambda_weights = np.ones([2, train_epoch], dtype=np.float32)
############################### model training
with strategy.scope():
    #opt = RectifiedAdam(lr=5e-3, min_lr=0, 
    #    warmup_proportion=0.1, total_steps=train_steps)
    opt = tf.keras.optimizers.Adam(lr=1e-4)        
    loss_estimator = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    if pretrain:
        model = frame.build_mtan_R2unet(
            pretrained_weights=os.path.join(pretrain, 'train.hdf5'))
    else:  
        model = frame.build_mtan_R2unet()

    train_loss_epoch = []; val_loss_epoch = []
    track_loss = np.inf
    st_epoch_idx = 1

    model.compile(optimizer = opt, loss=loss_estimator, metrics=['loss'])

    # Training loop
    for i in range(train_epoch):
        logging.info(f"Training epoch : {i+1}/{train_epoch}")
        if save_model_per_epoch:
            outdir_epoch = os.path.join(outdir, f'epoch_{i+1:04}')
            if not os.path.exists(outdir_epoch): 
                os.makedirs(outdir_epoch)
            save_dir = outdir_epoch
        else:
            save_dir = outdir
        train_list = np.random.permutation(train_list) 
        val_list = np.random.permutation(val_list)
        #stop

        ## temporary loss in each epoch
        if i==0 or i ==1:
            lambda_weights[:, i] = 1
        else:
            w_pick = avg_cost[i-1, 0] / avg_cost[i-2, 0]
            w_mask = avg_cost[i-1, 1] / avg_cost[i-2, 1]
            lambda_weights[0, i] = 2 * np.exp(w_pick / T) / \
                (np.exp(w_pick / T) + np.exp(w_mask / T))
            lambda_weights[1, i] = 2 * np.exp(w_mask / T) / \
                (np.exp(w_pick / T) + np.exp(w_mask / T))
        label_mask_weight = lambda_weights[:, i]
        logging.info(f"Pick weighting:{label_mask_weight[0]:.2f}\t"
            f"Mask weighting:{label_mask_weight[1]:.2f}")

        # training/validation steps per epoch
        n_train_data = len(train_list)
        n_val_data = len(val_list)
        train_steps_per_epoch = int(np.floor(n_train_data/global_batch_size))
        val_steps_per_epoch = int(np.floor(n_val_data/global_batch_size))

        # make and shuffle input data list
        train_list = np.random.permutation(train_list)
        val_list = np.random.permutation(val_list)
        # make data generator
        dis_train = iter(strategy.experimental_distribute_dataset(
            tfrecord_dataset_detect(train_list, data_length=data_length,
                 batch_size=global_batch_size)))
        dis_val = iter(strategy.experimental_distribute_dataset(
            tfrecord_dataset_detect(val_list, data_length=data_length,
                 batch_size=global_batch_size)))

        total_train_loss = 0
        total_loss_label = 0
        total_loss_mask = 0
        train_num_batches = 0
        for train_step in range(train_steps_per_epoch):
            # update model weights per step
            dis_train_trc_in, dis_train_label_in,\
                     dis_train_mask_in, dis_train_idx = next(dis_train)
            mean_train_batch_loss, per_replica_losses,\
                mean_batch_loss_label, mean_batch_loss_mask = \
                distributed_train_step_dwa(
                    strategy,
                    (model, opt, global_batch_size, 
                    dis_train_trc_in, dis_train_label_in,
                    dis_train_mask_in,
                    loss_estimator,
                    True,
                    label_mask_weight),)

            train_loss = tf.reduce_mean(mean_train_batch_loss)/\
                strategy.num_replicas_in_sync
            train_loss_label = tf.reduce_mean(mean_batch_loss_label)/\
                strategy.num_replicas_in_sync                
            train_loss_mask = tf.reduce_mean(mean_batch_loss_mask)/\
                strategy.num_replicas_in_sync  

            total_loss_label += train_loss_label.numpy()
            total_loss_mask += train_loss_mask.numpy()
            total_train_loss += train_loss.numpy()
            train_num_batches += 1
            # progress bar
            progbar = tf.keras.utils.Progbar(
                train_steps_per_epoch, interval=0.1,
                stateful_metrics=['step'])
            progbar.update(train_step+1)

        avg_cost[i, 0] += total_loss_label / train_num_batches
        avg_cost[i, 1] += total_loss_mask / train_num_batches
        train_loss_f = total_train_loss / train_num_batches

        total_val_loss = 0
        val_num_batches = 0
        for val_step in range(val_steps_per_epoch):
            # estimate validation dataset
            dis_val_trc_in, dis_val_label_in,\
                dis_val_mask_in, dis_val_idx = next(dis_val)
            mean_val_batch_loss = distributed_val_step_dwa(
                strategy, 
                (model, global_batch_size,
                dis_val_trc_in, dis_val_label_in, dis_val_mask_in,
                loss_estimator, label_mask_weight)
            )
            val_loss = tf.reduce_mean(mean_val_batch_loss)/\
                strategy.num_replicas_in_sync

            total_val_loss += val_loss.numpy()
            val_num_batches += 1
        val_loss_f = total_val_loss / val_num_batches 
        #stop
        
        progbar.update(train_step+1, 
            values=(
                ('train_loss',  train_loss_f),
                ('val_loss', val_loss_f)  
            ))

        train_loss_epoch.append(train_loss_f)
        val_loss_epoch.append(val_loss_f)
        # trace model improvements
        track_item = val_loss_f
        if  track_item < track_loss:
            message = save_model(
                model, save_dir, train_loss_epoch, 
                val_loss_epoch)
                
            logging.info("val_loss improved from "
                f"{track_loss:.6f} to {track_item:.6f}, {message}\n")
            track_loss = track_item
            ct_not_improve = 0
        else:
            ct_not_improve += 1
            logging.info(f"val_loss {track_item:.6f} "
                f"did not improve from {track_loss:.6f}"
                f" for {ct_not_improve} epochs.\n")

        if ct_not_improve == not_improve_patience:
            logging.info("Performance has not improved for "
                f"{ct_not_improve} epochs, stop training.")
            logging.info(f'Do you want to keep training? [yes/no]')
            decision = input().lower()
            if decision == 'yes':
                ct_not_improve = 0
                pass
            else:
                break