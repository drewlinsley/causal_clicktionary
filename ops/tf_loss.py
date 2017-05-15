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


def batchnorm(layer):
    m, v = tf.nn.moments(layer, [0])
    return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)


def choose_classifier(sample_layer, y, config):
    if config.classifier == 'softmax':
        weights, bias, preds = build_softmax(sample_layer)
        classifier, loss = softmax_optimization(
            yhat=preds, y=y, W=weights, b=bias, c=config.c, lr=config.lr)
    elif config.classifier == 'svm':
        weights, bias, preds = build_svm(sample_layer)
        classifier, loss = svm_optimization(
            yhat=preds, y=y, W=weights, b=bias, c=config.c, lr=config.lr)
    print 'Using a %s' % config.classifier
    return weights, preds, classifier, loss


def build_softmax(x):
    features = int(x.get_shape()[-1])
    W = tf.get_variable(
        'softmax_W', initializer=tf.zeros([features, 1]))
    b = tf.get_variable(
        'softmax_b', initializer=tf.zeros([1]))
    return W, b, tf.nn.bias_add(tf.matmul(x, W), b)


def softmax_optimization(yhat, y, W, b, c=0.01, lr=0.01):
    class_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(yhat, y))
    class_loss += (tf.nn.l2_loss(W) * c)  # l2 regularization
    return tf.train.GradientDescentOptimizer(lr).minimize(
        class_loss, var_list=[W, b]), class_loss


def build_svm(x):
    features = int(x.get_shape()[-1])
    W = tf.get_variable('svm_W', initializer=tf.zeros([features, 1]))
    b = tf.get_variable('svm_b', initializer=tf.zeros([1]))
    bnx = batchnorm(x)
    return W, b, tf.nn.bias_add(tf.matmul(bnx, W), b)


def svm_optimization(yhat, y, W, b, c=1, lr=0.01):
    regularization_loss = 0.5*tf.reduce_sum(tf.square(W))
    class_loss = hinge_loss(yhat, y)
    svm_loss = regularization_loss + c*class_loss
    return tf.train.GradientDescentOptimizer(lr).minimize(
        svm_loss, var_list=[W, b]), svm_loss

