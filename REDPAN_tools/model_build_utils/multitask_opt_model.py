import tensorflow as tf
import numpy as np
'''
Sources:
https://www.pyimagesearch.com/2020/03/23/using-tensorflow-and-gradienttape-to-train-a-keras-model/

# Encapsulating the forward and backward passs of data using
# tf.GradientTape for updating model weights.

'''

@tf.function
def train_step(model, loss_estimator, acc_estimator, 
                opt, train_trc, train_label, train_mask,
                training=True):
    # compute loss for gradient descent
    with tf.GradientTape() as tape:
        # make predictions and estimate loss
        train_pred_label, train_pred_mask = \
                model(train_trc, training=training)
        # loss
        train_loss_label = \
                loss_estimator(train_label, train_pred_label)
        train_loss_mask = \
                loss_estimator(train_mask, train_pred_mask)   
        # accuracy
        train_acc_label = \
                acc_estimator(train_label, train_pred_label)
        train_acc_mask = \
                acc_estimator(train_mask, train_pred_mask)   

        # summary
        train_loss = (train_loss_label + train_loss_mask)/2 
        train_acc = (train_acc_label + train_acc_mask)/2
    # calculate the gradients and update the weights
    grads = tape.gradient(train_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return train_loss, train_acc

@tf.function
def val_step(model, loss_estimator, acc_estimator,
             val_trc, val_label, val_mask, training=False):
    # make predictions and estimate loss
    val_pred_label, val_pred_mask = model(val_trc, training=training)
    # loss
    val_loss_label = \
            loss_estimator(val_label, val_pred_label)
    val_loss_mask = \
            loss_estimator(val_mask, val_pred_mask)   
    # accuracy
    val_acc_label = \
            acc_estimator(val_label, val_pred_label)
    val_acc_mask = \
            acc_estimator(val_mask, val_pred_mask)   

    val_loss = (val_loss_label + val_loss_mask)/2
    val_acc = (val_acc_label + val_acc_mask)/2
    return val_loss, val_acc

#--  distributed training
@tf.function
def distributed_train_step(strategy, train_args):
    def train_step_gpus(model, opt, global_batch_size,
            train_trc, train_label, train_mask, 
            loss_estimator, training=True, 
            label_mask_weight=[0.5, 0.5]):
        # compute loss for gradient descent
        with tf.GradientTape() as tape:
        # make predictions and estimate loss
            train_pred_label, train_pred_mask =\
                model(train_trc, training=True)
            per_replica_losses_label = \
                loss_estimator(train_label, train_pred_label)
            per_replica_losses_mask = \
                loss_estimator(train_mask, train_pred_mask)
            per_replica_losses = \
                per_replica_losses_label*label_mask_weight[0] + \
                    per_replica_losses_mask*label_mask_weight[1]

            grad_loss = tf.reduce_sum(per_replica_losses)\
                        /global_batch_size

        # calculate the gradients and update the weights
        grad = tape.gradient(grad_loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return per_replica_losses

    per_replica_losses = strategy.run(
                        train_step_gpus, args=train_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
                    per_replica_losses, axis=None)                               

    #train_loss_avg.update_state(mean_loss)
    return mean_batch_loss, per_replica_losses

@tf.function
def distributed_val_step(strategy, val_args):
    def val_step_gpus(
            model, global_batch_size,
            val_trc, val_label, val_mask, loss_estimator,
            label_mask_weight=[0.5, 0.5]):
        # estimate validation data 
        val_pred_label, val_pred_mask =\
                 model(val_trc, training=False)

        per_replica_losses_label =\
         loss_estimator(val_label, val_pred_label)
        per_replica_losses_mask =\
         loss_estimator(val_mask, val_pred_mask)
        per_replica_losses = \
            per_replica_losses_label*label_mask_weight[0] + \
                per_replica_losses_mask*label_mask_weight[1]

        return per_replica_losses
    per_replica_losses = strategy.run(val_step_gpus, args=val_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
                    per_replica_losses, axis=None)
    return mean_batch_loss

loss_mask = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
#--  distributed training
@tf.function
def distributed_train_step_dwa(strategy, train_args):
    def train_step_gpus(model, opt, global_batch_size,
            train_trc, train_label, train_mask, 
            loss_estimator, training=True, 
            label_mask_weight=[0.5, 0.5]):
        # compute loss for gradient descent
        with tf.GradientTape() as tape:
        # make predictions and estimate loss
            train_pred_label, train_pred_mask =\
                model(train_trc, training=True)
                
            per_replica_losses_label = \
                loss_estimator(train_label, train_pred_label)
            per_replica_losses_mask = \
                loss_estimator(train_mask, train_pred_mask)
            #per_replica_losses_mask = \
            #    loss_mask(train_mask, train_pred_mask)                
            per_replica_losses = \
                per_replica_losses_label*label_mask_weight[0] + \
                    per_replica_losses_mask*label_mask_weight[1]
            
            grad_loss = tf.reduce_sum(per_replica_losses)\
                        /global_batch_size

        # calculate the gradients and update the weights
        grad = tape.gradient(grad_loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return per_replica_losses, per_replica_losses_label,\
            per_replica_losses_mask

    per_replica_losses, per_replica_losses_label,\
            per_replica_losses_mask = strategy.run(
                        train_step_gpus, args=train_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses, axis=None)
    mean_batch_loss_label = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_label, axis=None)
    mean_batch_loss_mask = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_mask, axis=None)                    
                                 
    #train_loss_avg.update_state(mean_loss)
    return mean_batch_loss, per_replica_losses, \
        mean_batch_loss_label, mean_batch_loss_mask

@tf.function
def distributed_val_step_dwa(strategy, val_args):
    def val_step_gpus(
            model, global_batch_size,
            val_trc, val_label, val_mask, loss_estimator,
            label_mask_weight=[0.5, 0.5]):
        # estimate validation data 
        val_pred_label, val_pred_mask =\
                 model(val_trc, training=False)

        per_replica_losses_label =\
         loss_estimator(val_label, val_pred_label)
        per_replica_losses_mask =\
         loss_estimator(val_mask, val_pred_mask)
        #per_replica_losses_mask =\
        # loss_mask(val_mask, val_pred_mask)         
        per_replica_losses = \
            per_replica_losses_label*label_mask_weight[0] + \
                per_replica_losses_mask*label_mask_weight[1]

        return per_replica_losses
    per_replica_losses = strategy.run(val_step_gpus, args=val_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
                    per_replica_losses, axis=None)
    return mean_batch_loss


if __name__=='__main__':
    pass