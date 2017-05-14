import re
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from scipy import misc


def read_and_proc_images(image_names):
    label = []
    for idx, x in enumerate(image_names):
        im = misc.imread(x)
        if len(im.shape) == 2:
            im = np.repeat(im[:, :, None], 3, axis=-1)
        if im.shape[-1] > 3:
            im = im[:, :, :3]
        if idx == 0:
            images = im[None, :, :, :]
        else:
            images = np.append(images, im[None, :, :, :], axis=0)
        label = np.append(
            label, int(re.search('(?<=\/)(\d+)(?=\_)', x).group()))
    return images, label


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    This function is taken from keras backend
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis, x_shape[axis], x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis, x_rep)


def read_and_decode_single_example(
        filename,
        im_size,
        model_input_shape,
        data_augmentations=None,
        filename_key='filename/string',
        image_key='image/string',
        label_key='label/int64'):

    """first construct a queue containing a list of filenames.
    this lets a user split up there dataset in multiple files to keep
    size down"""
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=1)
    _, serialized_example = tf.TFRecordReader().read(filename_queue)
    keys = [k for k in [image_key, label_key, filename_key] if k is not None]
    fdict = get_tf_dict(keys)
    features = tf.parse_single_example(serialized_example, features=fdict)

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features[image_key.split('/')[0]], tf.float32)

    # Need to reconstruct channels first then transpose channels
    image = tf.reshape(image, im_size)  # np.asarray(im_size)[[2, 0, 1]])

    # image = tf.transpose(res_image, [2, 1, 0])
    image.set_shape(im_size)
    image = image_augmentations(
        image=image,
        heatmap=None,
        im_size=im_size,
        data_augmentations=data_augmentations,
        model_input_shape=model_input_shape,
        return_heatmaps=False)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features[label_key.split('/')[0]], tf.int32)
    return image, label


def get_crop_coors(image_size, target_size):
    h_diff = image_size[0] - target_size[0]
    ts = tf.constant(
        target_size[0], shape=[2, 1])
    offset = tf.cast(
        tf.round(tf.random_uniform([1], minval=0, maxval=h_diff)), tf.int32)
    return offset, ts[0], offset, ts[1]


def slice_op(image_slice, h_min, w_min, h_max, w_max):
    return tf.slice(
        image_slice, tf.cast(
            tf.concat(0, [h_min, w_min]), tf.int32), tf.cast(
            tf.concat(0, [h_max, w_max]), tf.int32))


def apply_crop(image, target, h_min, w_min, h_max, w_max):
    im_size = image.get_shape()
    if len(im_size) > 2:
        channels = []
        for idx in range(int(im_size[-1])):
            channels.append(
                slice_op(image[:, :, idx], h_min, w_min, h_max, w_max))
        out_im = tf.pack(channels, axis=2)
        out_im.set_shape([target[0], target[1], int(im_size[-1])])
        return out_im
    else:
        out_im = slice_op(image, h_min, w_min, h_max, w_max)
        return out_im.set_shape([target[0], target[1]])


def image_augmentations(
        image,
        heatmap,
        im_size,
        data_augmentations,
        model_input_shape,
        return_heatmaps):

    # Insert augmentation and preprocessing here
    if data_augmentations is not None:
        if 'random_crop' in data_augmentations:
            h_min, h_max, w_min, w_max = get_crop_coors(
                image_size=im_size, target_size=model_input_shape)
            image = apply_crop(
                image, model_input_shape, h_min, w_min, h_max, w_max)
            if return_heatmaps:
                heatmap = apply_crop(
                    heatmap, model_input_shape, h_min, w_min, h_max, w_max)
        elif 'resize' in data_augmentations:
            image = tf.squeeze(tf.image.resize_bilinear(
                tf.expand_dims(image, 0), model_input_shape[0:2]))
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, model_input_shape[0], model_input_shape[1])
            if return_heatmaps:
                heatmap = tf.image.resize_image_with_crop_or_pad(
                    heatmap, model_input_shape[0], model_input_shape[1])

        if 'left_right' in data_augmentations:
            lorr = tf.less(tf.random_uniform([], minval=0, maxval=1.), .5)
            image = control_flow_ops.cond(
                lorr,
                lambda: tf.image.flip_left_right(image),
                lambda: image)
            if return_heatmaps:
                heatmap = control_flow_ops.cond(
                    lorr,
                    lambda: tf.image.flip_left_right(heatmap),
                    lambda: heatmap)
        if 'random_contrast' in data_augmentations:
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        if 'random_brightness' in data_augmentations:
            image = tf.image.random_brightness(image, max_delta=63.)
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, model_input_shape[0], model_input_shape[1])

    # Make sure to clip values to [0, 1]
    image = image / 255.
    image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 1.0)

    if return_heatmaps:
        heatmap = tf.clip_by_value(tf.cast(heatmap, tf.float32), 0.0, 1.0)
        # Only take 1 slice from the heatmap
        heatmap = tf.expand_dims(heatmap[:, :, 0], dim=2)
        return image, heatmap
    else:
        return image


def interpret_key(k):
    if k == 'string':
        return tf.string
    elif k == 'int64':
        return tf.int64
    else:
        raise RuntimeError('Can\'t understand your key')


def get_tf_dict(keys):
    return {k.split('/')[0]: tf.FixedLenFeature(
        [], interpret_key(k.split('/')[-1])) for k in keys}


def read_and_decode(
        filename_queue,
        im_size,
        model_input_shape,
        data_augmentations=None,
        filename_key='filename/string',
        image_key='image/string',
        label_key='label/int64'):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    keys = [k for k in [image_key, label_key, filename_key] if k is not None]
    fdict = get_tf_dict(keys)
    features = tf.parse_single_example(serialized_example, features=fdict)

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features[image_key.split('/')[0]], tf.float32)

    # Need to reconstruct channels first then transpose channels
    image = tf.reshape(image, im_size)  # np.asarray(im_size)[[2, 0, 1]])

    # image = tf.transpose(res_image, [2, 1, 0])
    image.set_shape(im_size)
    # image = tf.expand_dims(image, 0)
    image = image_augmentations(
        image=image,
        heatmap=None,
        im_size=im_size,
        data_augmentations=data_augmentations,
        model_input_shape=model_input_shape,
        return_heatmaps=False)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features[label_key.split('/')[0]], tf.int32)

    # Get filename as a string.
    filename = tf.cast(features[filename_key.split('/')[0]], tf.string)
    return image, label, filename


def inputs(
        tfrecord_file,
        batch_size,
        im_size,
        model_input_shape,
        num_epochs=None,
        num_threads=2,
        data_augmentations=None,
        shuffle_batch=True):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        batch_data = read_and_decode(
            filename_queue,
            im_size,
            model_input_shape,
            data_augmentations=data_augmentations)
        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if shuffle_batch:
            images, labels, filenames = tf.train.shuffle_batch(
                batch_data, batch_size=batch_size, num_threads=num_threads,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
        else:
            images, labels, filenames = tf.train.batch(
                batch_data, batch_size=batch_size, num_threads=num_threads,
                capacity=1000 + 3 * batch_size)

        return images, labels, filenames
