import os
import time
import re
import sys
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from ops.data_loader import inputs
from settings import config
from models import baseline_vgg16 as vgg16
from glob import glob
from ops.tf_loss import class_accuracy


# Train or finetune a vgg16 while cuing to clickme
def test_vgg16():
    dbc = config.config()

    validation_pointer = os.path.join(
        dbc.packaged_data_path, '%s_%s.%s' % (
            'validation',
            dbc.packaged_data_file,
            dbc.output_format))

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        val_images, val_labels, val_files = inputs(
            tfrecord_file=validation_pointer,
            batch_size=dbc.validation_batch,
            im_size=dbc.validation_image_size,
            model_input_shape=dbc.model_image_size[:2],
            num_epochs=1,
            data_augmentations=dbc.validation_augmentations,
            shuffle_batch=True)

    # Prepare pretrained model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            cnn = vgg16.Vgg16()
            validation_mode = tf.Variable(False, name='training')
            cnn.build(
                val_images, output_shape=1000,
                train_mode=validation_mode)
            sample_layer = cnn['fc7']
            accs = class_accuracy(cnn.prob, val_labels)
    saver = tf.train.Saver(
        tf.all_variables(), max_to_keep=10)

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.group(tf.initialize_all_variables(),
             tf.initialize_local_variables()))
    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver.restore(sess, dbc.model_types['vgg16'][0])
    # Start training loop
    results = {
        'accs': [],
        'preds': [],
        'labs': [],
        'files': []
    }
    np_path = os.path.join(
        dbc.checkpoint_directory, 'validation_results')
    step = 0
    scores, labels = [], []
    try:
        print 'Testing model'
        while not coord.should_stop():
            start_time = time.time()
            score, lab, f, probs  = sess.run(
                [sample_layer, val_labels, val_files, cnn['prob']])
            import ipdb;ipdb.set_trace()
            print acc

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


if __name__ == '__main__':
    test_vgg16()


