import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from ops.data_loader import inputs
from ops import tf_loss
import cPickle
from tqdm import tqdm


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


def test_classifier(
        validation_pointer,
        model_ckpt,
        model_dir,
        model_type,
        model_weights,
        selected_layer,
        config,
        simulate_subjects=120):

    # Make output directories if they do not exist
    config.checkpoint_directory = model_dir

    print '-'*60
    print'Testing the model over a %s. Saving to %s' % (model_type, model_dir)
    print '-'*60

    dcn_flavor = import_cnn(model_type)

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        val_images, val_labels, val_files = inputs(
            tfrecord_file=validation_pointer,
            batch_size=config.validation_batch,
            im_size=config.validation_image_size,
            model_input_shape=config.model_image_size[:2],
            num_epochs=1,
            data_augmentations=config.validation_augmentations,
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
                val_images)
            sample_layer = cnn[selected_layer]


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

    with open(model_ckpt, 'rb') as fid:
        svc = cPickle.load(fid)
        zscorer = np.load(model_ckpt.split('.')[0] + '_normalization.npz')
    np_path = os.path.join(
        config.checkpoint_directory, 'validation_results')
    step = 0
    scores, labels = [], []
    if '.ckpt' in model_weights:
        saver.restore(sess, model_weights) 
        print 'Restored model from %s' % model_weights
    try:
        print 'Testing model'
        while not coord.should_stop():
            start_time = time.time()
            score, lab, f = sess.run(
                [sample_layer, val_labels, val_files])
            norm_score = (score - zscorer['mu']) / zscorer['sd']
            scores += [norm_score]
            pred = svc.predict(norm_score)
            acc = np.mean(pred == lab)
            results['accs'] += [acc]
            results['preds'] += [pred]
            results['labs'] += [lab]
            results['files'] += [f]
            duration = time.time() - start_time
            print_status(step, 0, config, duration, acc, np_path)
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

    if simulate_subjects:
        sim_subs = []
        print 'Simulating subjects'
        scores = np.concatenate(scores)
        labels = np.concatenate(results['labs'])
        for sub in tqdm(range(simulate_subjects)):
            it_results = {
                'accs': [],
                'preds': [],
                'labs': [],
                'files': []
            }

            neuron_drop = np.random.rand(scores.shape[1]) > .95
            it_scores = np.copy(scores)
            it_scores[:, neuron_drop] = 0
            pred = svc.predict(it_scores)
            acc = np.mean(pred == labels)
            it_results['accs'] += [acc]
            it_results['preds'] += [pred]
            it_results['labs'] += [labels]
            it_results['files'] += [np.concatenate(results['files'])]
            sim_subs += [it_results]
        np.save(np_path + '_sim_subs', sim_subs)
