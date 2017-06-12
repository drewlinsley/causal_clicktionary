#!/usr/bin/env python
import os
import sys
import tensorflow as tf
from ops import data_loader
from argparse import ArgumentParser
from timeit import default_timer as timer
from matplotlib import pyplot as plt


def prepare_data(filename, size, config):
    label, image = data_loader.read_and_decode_single_example(
        filename=filename,
        im_size=size,
        model_input_shape=config.model_image_size[:2])
    return label, image


def test_tf_record(sess, labels, images, plot, num_tests):
    # first example from file
    for idx in range(num_tests):
        start = timer()
        label_val, image_val = sess.run([labels, images])
        # import ipdb;ipdb.set_trace()
        # plt.imshow(image_val); plt.show()
        delta = timer() - start
        sys.stdout.write(
            '\rEntry %s extracted in: %s seconds' % (idx, delta)),
        sys.stdout.flush()
        if plot:
            f, ax = plt.subplots(1)
            ax.imshow(image_val)
            ax.set_title('Label: %s' % label_val)
            plt.show()
            plt.close(f)


def main(config_name=None, num_tests=10, plot=False):
    """
    Runs a query on the database and returns an image list that can be
    packaged into a portable file format.
    :::
    Inputs: config_name, the file used to generate a labeling dataset.
    Each file will become a different category.
    """
    if config_name is None:
        raise RuntimeError(
            'You need to pass the name of your label config file!')
    in_config = getattr(
        __import__('settings', fromlist=[config_name]), config_name)
    dbc = in_config.config()

    # Always train on "train" tfrecords and test on validation
    tf_pointers = {k: os.path.join(
        dbc.packaged_data_path, '%s_%s.%s' % (
            k,
            dbc.packaged_data_file,
            dbc.output_format)) for k in dbc.package_indices}
    train_pointer = [v for k, v in tf_pointers.iteritems() if 'train' in k][0]
    train_label, train_image = prepare_data(
        filename=train_pointer,
        size=dbc.train_image_size,
        config=dbc)
    validation_pointer = [v for k, v in tf_pointers.iteritems() if 'val' in k][0]
    validation_label, validation_image = prepare_data(
        filename=validation_pointer,
        size=dbc.validation_image_size,
        config=dbc)

    # Start session
    sess = tf.Session()
    sess.run(
        tf.group(
            tf.initialize_all_variables(), tf.initialize_local_variables()))
    tf.train.start_queue_runners(sess=sess)

    # Train
    test_tf_record(
        sess=sess,
        labels=train_label,
        images=train_image,
        plot=plot,
        num_tests=num_tests)

    # validation
    test_tf_record(
        sess=sess,
        labels=train_label,
        images=train_image,
        plot=plot,
        num_tests=num_tests)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, dest="config_name",
        default='config', help="Name of labeling configuration file.")
    args = parser.parse_args()
    main(**vars(args))
