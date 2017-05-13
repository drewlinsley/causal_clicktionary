import os
import sys
import tensorflow as tf
from ops import data_loader
from timeit import default_timer as timer
from matplotlib import pyplot as plt


def run_tester(config_name='config', which_set='validation'):
    if config_name is None:
        raise RuntimeError(
            'You need to pass the name of your label config file!')
    in_config = getattr(
        __import__('settings', fromlist=[config_name]), config_name)
    dbc = in_config.config()
    if which_set == 'train':
        image_size = dbc.train_image_size
        augmentations = dbc.train_augmentations
    elif which_set == 'validation':
        image_size = dbc.validation_image_size
        augmentations = dbc.validation_augmentations
    else:
        raise RuntimeError('Cannot interpret your CV set')
    tf_pointer = os.path.join(
        dbc.packaged_data_path, '%s_%s.%s' % (
            which_set,
            dbc.packaged_data_file,
            dbc.output_format))
    images, labels = data_loader.read_and_decode_single_example(
        tf_pointer,
        image_size,
        dbc.model_image_size[:2],
        data_augmentations=augmentations)
    images, labels = tf.train.shuffle_batch(
        [images, labels], batch_size=1, num_threads=1,
        capacity=1000 + 3,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    sess = tf.Session()

    # Required. See below for explanation
    sess.run(tf.group(tf.initialize_all_variables(),
             tf.initialize_local_variables()))
    tf.train.start_queue_runners(sess=sess)

    # first example from file
    count = 0
    while 1:
        start = timer()
        it_im, it_lab = sess.run([images, labels])
        delta = timer() - start
        sys.stdout.write(
            '\rEntry %s extracted in: %s seconds' % (count, delta)),
        sys.stdout.flush()
        count += 1
        f = plt.figure()
        plt.subplot(1, 1, 1)
        plt.imshow(it_im[0,:,:,:])
        plt.title(
            'Image size: %s, %s, %s' % (
                it_im.shape[0], it_im.shape[1], it_im.shape[2]))
        plt.title('Category: %s' % it_lab)
        plt.show()
        plt.close(f)
        if it_lab is None:
            sys.stdout.write('\n')
            break


if __name__ == '__main__':
    run_tester()
