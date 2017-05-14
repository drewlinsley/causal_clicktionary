import tensorflow as tf


def fine_tune_prepare_layers(tf_vars, finetune_vars):
    ft_vars = []
    other_vars = []
    for v in tf_vars:
        ss = [v.name.find(x) != -1 for x in finetune_vars]
        if True in ss:
            ft_vars.append(v)
        else:
            other_vars.append(v)
    return other_vars, ft_vars


def finetune_learning(
        loss,
        lr,
        trainable_variables,
        fine_tune_layers=None,
        optimizer='adam'):
    if fine_tune_layers is not None:
        hold_lr = lr / 3
        other_opt_vars, ft_opt_vars = fine_tune_prepare_layers(
            trainable_variables, fine_tune_layers)
        if optimizer == 'adam':
            train_op, gvs = ft_optimizer_list(
                loss, [other_opt_vars, ft_opt_vars],
                tf.train.AdamOptimizer,
                [hold_lr, lr])
        elif optimizer == 'sgd':
            train_op, gvs = ft_optimizer_list(
                loss, [other_opt_vars, ft_opt_vars],
                tf.train.GradientDescentOptimizer,
                [hold_lr, lr])
    else:
        if optimizer == 'adam':
            train_op = tf.train.AdamOptimizer(
                lr).minimize(loss)
        elif optimizer == 'sgd':
            train_op = tf.train.GradientDescentOptimizer(
                lr).minimize(loss)
    return train_op


def ft_optimizer_list(cost, opt_vars, optimizer, lrs, grad_clip=False):
    """Efficient optimization for fine tuning a net."""
    ops = []
    gvs = []
    for v, l in zip(opt_vars, lrs):
        if grad_clip:
            optim = optimizer(l)
            gvs = optim.compute_gradients(cost, var_list=v)
            capped_gvs = [
                (tf.clip_by_norm(grad, 10.), var)
                if grad is not None else (grad, var) for grad, var in gvs]
            ops.append(optim.apply_gradients(capped_gvs))
        else:
            ops.append(optimizer(l).minimize(cost, var_list=v))
    return tf.group(*ops), gvs


def class_accuracy(pred, targets):
    # assuming targets is an index
    return tf.reduce_mean(
        tf.to_float(tf.equal(tf.argmax(pred, 1), tf.cast(
            targets, dtype=tf.int64))))


def hinge_loss(yhat, y):
    return tf.reduce_mean(tf.nn.relu(
        1 - (tf.squeeze(yhat) * tf.cast(y, tf.float32))))
    # return tf.contrib.losses.hinge_loss(tf.squeeze(yhat), tf.cast(y, tf.float32))
