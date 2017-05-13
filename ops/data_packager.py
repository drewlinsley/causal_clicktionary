import traceback
import sys
import os
import json
import numpy as np
from scipy import misc
import tensorflow as tf
from ops.utilities import make_dir
from tqdm import tqdm
from copy import deepcopy


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def load_image(file):
    im = misc.imread(file)
    if len(im.shape) < 3:
        im = np.repeat(im[:, :, None], 3, axis=-1)
    return im


def image_converter(im_ext):
    if im_ext == '.jpg' or im_ext == '.jpeg' or im_ext == '.JPEG':
        out_fun = tf.image.encode_jpeg
    elif im_ext == '.png':
        out_fun = tf.image.encode_png
    else:
        print '-'*60
        traceback.print_exc(file=sys.stdout)
        print '-'*60
    return out_fun


def apply_tf_feature(op, data):
    if op == 'int64':
        return int64_feature(int(data))
    elif op == 'string':
        return bytes_feature(data)


def create_example(data_dict):
    return {k.split('/')[0]: apply_tf_feature(
        k.split('/')[1], v) for k, v in data_dict.iteritems()}


def encode_tf_record(data_dict):
    # construct the Example proto boject
    return tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features has a map of string to Feature proto objects
            feature=create_example(data_dict)
        )
    )


def extract_to_tf_records(
        data_dict,
        output_pointer,
        file_path = '',
        filename_key='filename/string',
        image_key='image/string'):
    print 'Building: %s' % output_pointer
    with tf.python_io.TFRecordWriter(output_pointer) as tfrecord_writer:
        for row_dict in tqdm(data_dict):
            row_dict[image_key] = load_image(
                os.path.join(file_path, row_dict[filename_key])).astype(
                np.float32).tostring()
            if row_dict[image_key] is not None:
                record = encode_tf_record(row_dict)
                # use the proto object to serialize the example to a string
                serialized = record.SerializeToString()
                # write the serialized object to disk
                tfrecord_writer.write(serialized)


def tfrecords(data_dict, file_path, output_pointer):
    # Make dirs if they do not exist
    split_pointer = output_pointer.split('/')[:-1]
    if split_pointer[0] == '':
        split_pointer[0] = os.path.sep
    [make_dir(
        os.path.join(*split_pointer[:(idx + 1)])) for idx in range(
        len(split_pointer))]
    # Create tfrecords
    extract_to_tf_records(
        data_dict=deepcopy(data_dict),
        file_path=file_path,
        output_pointer=output_pointer)
    print 'Saved to %s' % output_pointer
    # Also save a "meta" info file
    with open(
            output_pointer.split('.tfrecords')[0] + '_meta.json', 'w') as fout:
        json.dump(data_dict, fout)
