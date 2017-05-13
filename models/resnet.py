import numpy as np
import tensorflow as tf

n_dict = {20:1, 32:2, 44:3, 56:4}
# ResNet architectures used for CIFAR-10
class model:
    """
    A trainable version of resnet.
    """

    def __init__(
                self,
                n=20,
                weight_path=None,
                trainable=True,
                fine_tune_layers=None):
        if weight_path is not None:
            self.data_dict = np.load(weight_path, encoding='latin1').item()
            # pop the specified keys from the weights that will be loaded
            if fine_tune_layers is not None:
                for key in fine_tune_layers:
                    del self.data_dict[key]
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.n = n
        self.layers = []

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(self,
              rgb, 
              output_shape=None,
              train_mode=None,
              batchnorm=None):

        if self.n < 20 or (self.n - 20) % 12 != 0:
            raise "ResNet depth invalid."

        if output_shape is None:
            output_shape = 1000

        rgb_scaled = rgb * 255.0  # Scale up to imagenet's uint8

        with tf.variable_scope('conv1'):
            conv1 = conv_layer(inpt, [3, 3, 3, 16], 1)
            layers.append(conv1)

        for i in range (num_conv):
            with tf.variable_scope('conv2_%d' % (i+1)):
                conv2_x = residual_block(layers[-1], 16, False)
                conv2 = residual_block(conv2_x, 16, False)
                layers.append(conv2_x)
                layers.append(conv2)

            assert conv2.get_shape().as_list()[1:] == [32, 32, 16]

        for i in range (num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv3_%d' % (i+1)):
                conv3_x = residual_block(layers[-1], 32, down_sample)
                conv3 = residual_block(conv3_x, 32, False)
                layers.append(conv3_x)
                layers.append(conv3)

            assert conv3.get_shape().as_list()[1:] == [16, 16, 32]
    
        for i in range (num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv4_%d' % (i+1)):
                conv4_x = residual_block(layers[-1], 64, down_sample)
                conv4 = residual_block(conv4_x, 64, False)
                layers.append(conv4_x)
                layers.append(conv4)

            assert conv4.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc'):
            global_pool = tf.reduce_mean(layers[-1], [1, 2])
            assert global_pool.get_shape().as_list()[1:] == [64]
        
            out = softmax_layer(global_pool, [64, 10])
            layers.append(out)

        return layers[-1]

    def weight_variable(shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def softmax_layer(inpt, shape):
        fc_w = weight_variable(shape)
        fc_b = tf.Variable(tf.zeros([shape[1]]))

        fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

        return fc_h

    def conv_layer(inpt, filter_shape, stride):
        out_channels = filter_shape[3]

        filter_ = weight_variable(filter_shape)
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
        mean, var = tf.nn.moments(conv, axes=[0,1,2])
        beta = tf.Variable(tf.zeros([out_channels]), name="beta")
        gamma = weight_variable([out_channels], name="gamma")
    
        batch_norm = tf.nn.batch_norm_with_global_normalization(
            conv, mean, var, beta, gamma, 0.001,
            scale_after_normalization=True)

        out = tf.nn.relu(batch_norm)

        return out

    def residual_block(inpt, output_depth, down_sample, projection=False):
        input_depth = inpt.get_shape().as_list()[3]
        if down_sample:
            filter_ = [1,2,2,1]
            inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

        conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
        conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

        if input_depth != output_depth:
            if projection:
                # Option B: Projection shortcut
                input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
            else:
                # Option A: Zero-padding
                input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
        else:
            input_layer = inpt

        res = conv2 + input_layer
        return res

