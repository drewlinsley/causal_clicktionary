import os
import re
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from ops import utilities
from ops.data_loader import inputs
from ops import tf_loss


def print_status(step, loss_value, config, duration, validation_accuracy):
    format_str = (
        '%s: step %d, loss = %.2f (%.1f examples/sec; '
        '%.3f sec/batch) | validation accuracy %.3f | logdir = %s')
    print (
        format_str % (
        datetime.now(),
        step,
        loss_value,
        config.train_batch / duration,
        validation_accuracy,
        float(duration)))


def import_cnn(model_type):
    return getattr(
        __import__('models', fromlist=[model_type]), model_type)


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
        validation_pointer,
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
    print'Training an SVM over a %s. Saving to %s' % (model_type, dt_stamp)
    print '-'*60

    dcn_flavor = import_cnn(model_type)

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_images, train_labels = inputs(
            train_pointer,
            config.train_batch,
            config.train_image_size,
            config.model_image_size[:2],
            num_epochs=config.epochs,
            shuffle_batch=True)
        val_images, val_labels = inputs(
            validation_pointer,
            config.validation_batch,
            config.validation_image_size,
            config.model_image_size[:2],
            num_epochs=None,
            shuffle_batch=False)

    # Prepare pretrained model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:
            cnn = dcn_flavor.model(
                weight_path=model_weights)
            cnn.build(
                train_images)
            sample_layer = cnn[selected_layer]  # sample features here with a mask: self.number_of_features
            svm_weights, svm_guesses = build_svm(
                sample_layer)
            svm_model, svm_loss = svm_optimization(
                yhat=svm_guesses,
                y=train_labels,
                W=svm_weights,
                c=config.svm_c,
                lr=config.lr)

            scope.reuse_variables()
            val_cnn = dcn_flavor.model(
                weight_path=model_weights)
            val_cnn.build(
                val_images)
            _, val_svm_guesses = build_svm(
                val_cnn[selected_layer])

            # Calculate validation accuracy
            val_accuracy = tf_loss.class_accuracy(tf.sign(val_svm_guesses), val_labels)
            tf.scalar_summary("validation accuracy", val_accuracy)

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
    np.save(
        os.path.join(
            config.checkpoint_directory, 'training_config_file'), config)
    step, losses = [], []
    results = {
        'accs': [],
        'preds': [],
        'labs': []
    }
    try:
        while not coord.should_stop():
            start_time = time.time()
            import ipdb;ipdb.set_trace()
            _, loss_value, images, labels = sess.run(
                [svm_model, svm_loss, train_images, train_labels])
            losses.append(loss_value)
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # Training status and validation accuracy
            acc, pred, lab = sess.run(
                [val_accuracy, val_svm_guesses, val_labels])
            results['accs'] += [acc]
            results['preds'] += [pred]
            results['labs'] += [lab]
            print_status(step, loss_value, config, duration, val_accuracy)

        # End iteration
        step += 1

    except tf.errors.OutOfRangeError:
        print 'Done training for %d epochs, %d steps.' % (config.epochs, step)
        print 'Saved to: %s' % config.checkpoint_directory
    finally:
        ckpt_path = os.path.join(
                config.checkpoint_directory,
                'model_' + str(step) + '.ckpt')
        saver.save(
            sess, ckpt_path, global_step=step)
        print 'Saved checkpoint to: %s' % ckpt_path
        coord.request_stop()
        np.savez(
            os.path.join(
                config.checkpoint_directory, 'validation_results'), **results)
    coord.join(threads)
    sess.close()
