import os
import re
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from ops.data_loader import inputs
from ops import tf_loss


def print_status(step, loss_value, config, duration, validation_accuracy, log_dir):
    format_str = (
        '%s: step %d, loss = %.2f (%.1f examples/sec; '
        '%.3f sec/batch) | validation accuracy %.3f | logdir = %s')
    print (
        format_str % (
            datetime.now(),
            step,
            loss_value,
            config.train_batch / duration,
            float(duration),
            validation_accuracy,
            log_dir))


def import_cnn(model_type):
    return getattr(
        __import__('models', fromlist=[model_type]), model_type)


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
    return W, tf.matmul(x, W) + b


def svm_optimization(yhat, y, W, c=1, lr=0.01):
    regularization_loss = 0.5*tf.reduce_sum(tf.square(W))
    class_loss = tf_loss.hinge_loss(yhat, y)
    svm_loss = regularization_loss + c*class_loss
    return tf_loss.finetune_learning(
        loss=svm_loss,
        trainable_variables=tf.trainable_variables(),
        lr=0,
        fine_tune_layers=['svm_W', 'svm_b'],
        optimizer='adam'), svm_loss


def test_classifier(
        validation_pointer,
        model_ckpt,
        model_dir,
        model_type,
        model_weights,
        selected_layer,
        config):

    # Make output directories if they do not exist
    config.checkpoint_directory = model_dir

    print '-'*60
    print'Testing the model over a %s. Saving to %s' % (model_type, model_dir)
    print '-'*60

    dcn_flavor = import_cnn(model_type)

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        val_images, val_labels, val_files = inputs(
            validation_pointer,
            config.validation_batch,
            config.validation_image_size,
            config.model_image_size[:2],
            1,
            shuffle_batch=False)

    # Prepare pretrained model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            if 'ckpt' in model_weights:
                cnn = dcn_flavor.model()
            else:
                cnn = dcn_flavor.model(
                    weight_path=model_weights)
            cnn.build(
                val_images)
            sample_layer = cnn[selected_layer]
            weights, yhat, classifier, class_loss = choose_classifier(
                sample_layer=sample_layer,
                y=val_labels,
                config=config)

    saver = tf.train.Saver(
        tf.all_variables(), max_to_keep=10)
    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.initialize_all_variables(),
             tf.initialize_local_variables()))
    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training loop
    results = {
        'accs': [],
        'preds': [],
        'labs': [],
        'files': []
    }
    if int(tf.__version__.split('.')[1]) > 10:
        model_ckpt += '-%s' % re.search(
            '\d+.ckpt', model_ckpt).group().split('.ckpt')[0]
    saver.restore(sess, model_ckpt)
    np_path = os.path.join(
        config.checkpoint_directory, 'validation_results')
    step = 0
    try:
        print 'Testing model'
        while not coord.should_stop():
            start_time = time.time()
            pred, lab, f = sess.run(
                [yhat, val_labels, val_files])
            import ipdb;ipdb.set_trace()
            results['accs'] += [np.mean(
                ((pred > 0).astype(float) == lab).astype(float))]
            results['preds'] += [pred]
            results['labs'] += [lab]
            results['files'] += [f]
            duration = time.time() - start_time
            print_status(step, 0, config, duration, results['accs'], np_path)
            step += 1

    except tf.errors.OutOfRangeError:
        print 'Done testing.'
    finally:
        np.savez(np_path, **results)
        print 'Saved to: %s' % np_path
        coord.request_stop()
    coord.join(threads)
    sess.close()
    print '%.4f%% correct' % np.mean(results['accs'])
