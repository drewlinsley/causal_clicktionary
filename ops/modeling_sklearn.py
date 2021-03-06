import os
import re
import time
import numpy as np
import tensorflow as tf
import cPickle
from datetime import datetime
from ops import utilities
from ops.data_loader import inputs
from ops import tf_loss
from sklearn import svm


def print_status(
        step, loss_value, config, duration, validation_accuracy, log_dir):
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
        weights, preds = build_softmax(sample_layer)
        classifier, loss = softmax_optimization(
            preds, y, weights, config.c, config.lr)
    elif config.classifier == 'svm':
        weights, preds = build_svm(sample_layer)
        classifier, loss = svm_optimization(
            preds, y, weights, config.c, config.lr)
    print 'Using a %s' % config.classifier
    return weights, preds, classifier, loss


def build_softmax(x):
    features = int(x.get_shape()[-1])
    W = tf.get_variable(
        'softmax_W', initializer=tf.zeros([features, 1]))
    b = tf.get_variable(
        'softmax_b', initializer=tf.zeros([1]))
    return W, tf.nn.bias_add(tf.matmul(x, W), b)


def softmax_optimization(yhat, y, W, c=0.01, lr=0.01):
    class_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(yhat, y))
    class_loss += (tf.nn.l2_loss(W) * c)  # l2 regularization
    return tf_loss.finetune_learning(
        loss=class_loss,
        trainable_variables=tf.trainable_variables(),
        lr=lr,
        fine_tune_layers=['softmax_W', 'softmax_b'],
        optimizer='adam'), class_loss


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
        lr=lr,
        fine_tune_layers=['svm_W', 'svm_b'],
        optimizer='adam'), svm_loss


def train_classifier_on_model(
        train_pointer,
        model_type,
        model_weights,
        selected_layer,
        config):

    # Make output directories if they do not exist
    dt_stamp = '%s_%s_%s_%s' % (
        model_type,
        selected_layer,
        str(config.lr)[2:],
        re.split(
            '\.', str(datetime.now(
                )))[0].replace(' ', '_').replace(':', '_').replace('-', '_')
        )
    config.checkpoint_directory = os.path.join(
        config.checkpoint_directory, dt_stamp)  # timestamp this run
    dir_list = [config.checkpoint_directory]
    [utilities.make_dir(d) for d in dir_list]

    print '-'*60
    print'Training %s over a %s. Saving to %s' % (
        config.classifier, model_type, dt_stamp)
    print '-'*60

    dcn_flavor = import_cnn(model_type)

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_images, train_labels, train_files = inputs(
            train_pointer,
            config.train_batch,
            config.train_image_size,
            config.model_image_size[:2],
            num_epochs=config.epochs,
            shuffle_batch=True)

    # Prepare pretrained model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            if 'ckpt' in model_weights:
                cnn = dcn_flavor.model()
            else:
                cnn = dcn_flavor.model(
                    weight_path=model_weights)
            cnn.build(
                train_images)
            sample_layer = cnn[selected_layer]
            class_accuracy = tf_loss.class_accuracy(cnn.prob, train_labels)

    saver = tf.train.Saver(
        tf.all_variables(), max_to_keep=10)

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(
        tf.group(
            tf.initialize_all_variables(), tf.initialize_local_variables()))
    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training loop
    np.save(
        os.path.join(
            config.checkpoint_directory, 'training_config_file'), config)
    step, scores, labs = 0, [], []
    if 'ckpt' in model_weights:
        saver.restore(sess, model_weights)
    try:
        print 'Getting scores'
        while not coord.should_stop():
            import ipdb;ipdb.set_trace()
            start_time = time.time()
            score, lab, acc = sess.run(
                [sample_layer, train_labels, class_accuracy])
            scores += [score]
            labs += [lab]
            duration = time.time() - start_time
            # End iteration
            print_status(step, 1, config, duration, acc, '')
            step += 1
    except tf.errors.OutOfRangeError:
        print 'Finished extracting scores.'
    finally:
        coord.request_stop()

    X = np.concatenate(scores)
    y = np.concatenate(labs)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    X = (X - mu) / sd
    svc = svm.LinearSVC(dual=False, C=config.c, verbose=True).fit(
        X, y)
    ckpt_path = os.path.join(
            config.checkpoint_directory,
            'model_%s.pkl' % step)
    with open(ckpt_path, 'wb') as fid:
        cPickle.dump(svc, fid)
    norm_path = os.path.join(
            config.checkpoint_directory,
            'model_%s_normalization' % step)
    np.savez(norm_path, mu=mu, sd=sd, scores=scores, labs=labs)
    print 'Saved to: %s' % config.checkpoint_directory
    print 'Saved checkpoint to: %s' % ckpt_path
    coord.join(threads)
    sess.close()
    # Return the final checkpoint for testing
    return ckpt_path, config.checkpoint_directory
