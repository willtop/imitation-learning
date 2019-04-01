from __future__ import print_function
import numpy as np
import tensorflow as tf
from local_settings import *


def weight_xavi_init(shape, name):
    initial = tf.get_variable(name=name, shape=shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    return initial


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class Network(object):

    def __init__(self):
        """ We put a few counters to see how many times we called each function """
        self._dropout_vec = [1.0] * 4 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
        self._amount_of_commands = 3 # [left, straight, right]
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._weights = {}
        self._features = {}
        self._learning_rate = 1e-4
        self.TFgraph = tf.Graph()

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_' + str(self._count_conv))
        bias = bias_variable([output_size], name='B_c_' + str(self._count_conv))

        self._weights['W_conv' + str(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)
        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_' + str(self._count_conv)), bias,
                          name='add_' + str(self._count_conv))

        self._features['conv_block' + str(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool' + str(self._count_pool))

    def bn(self, x, whether_training):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=whether_training,
                                            updates_collections=None,
                                            scope='bn' + str(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu' + str(self._count_activations))

    def dropout(self, x, whether_training):
        print("Dropout", self._count_dropouts)
        self._count_dropouts += 1
        output = tf.cond(whether_training,
                         lambda: tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1], name='dropout' + str(self._count_dropouts)),
                         lambda: tf.nn.dropout(x, 1, name='dropout' + str(self._count_dropouts)))
        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_' + str(self._count_fc))
        bias = bias_variable([output_size], name='B_f_' + str(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_' + str(self._count_fc))

    def conv_block(self, x, kernel_size, stride, output_size, whether_training, padding_in='SAME'):
        print(" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x, whether_training)
            x = self.dropout(x, whether_training)

            return self.activation(x)

    def fc_block(self, x, output_size, whether_training):
        print(" === FC", self._count_fc, "  :  ", output_size)
        with tf.name_scope("fc" + str(self._count_fc + 1)):
            x = self.fc(x, output_size)
            x = self.dropout(x, whether_training)
            self._features['fc_block' + str(self._count_fc + 1)] = x
            return self.activation(x)

    # Final sigmoid layer predicting the safety score for each possible command
    def fc_outputs(self, x):
        print(" === Final FC : ", self._amount_of_commands)
        with tf.name_scope("final_fc"):
            x = self.fc(x, self._amount_of_commands)
            return x

    def build_rejection_network(self):
        with self.TFgraph.as_default():
            input_images = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_images")
            targets = tf.placeholder(tf.float32, shape=[None, self._amount_of_commands], name="targets")
            whether_training = tf.placeholder(tf.bool, name="whether_it_is_training")

            """conv1"""  # kernel sz, stride, num feature maps
            xc = self.conv_block(input_images, 5, 2, 32, whether_training, padding_in='VALID')
            print(xc)
            xc = self.conv_block(xc, 3, 1, 32, whether_training, padding_in='VALID')
            print(xc)

            """conv2"""
            xc = self.conv_block(xc, 3, 2, 64, whether_training, padding_in='VALID')
            print(xc)
            xc = self.conv_block(xc, 3, 1, 64, whether_training, padding_in='VALID')
            print(xc)

            """ reshape """
            x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
            print(x)

            """ fc1 """
            x = self.fc_block(x, 128, whether_training)
            print(x)
            """ fc2 """
            x = self.fc_block(x, 64, whether_training)

            """ final layer computing safety score for each command"""
            safety_scores_logits = self.fc_outputs(x)

            """ Loss function (Weighted Cross Entropy to Penalize Largely on False Negative)"""
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets, logits=safety_scores_logits, name="Vanilla_CE_Loss"))
            """ Train step"""
            train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

            """ Final Safety Scores"""
            safety_scores = tf.nn.sigmoid(safety_scores_logits)

        return self.TFgraph, input_images, targets, whether_training, safety_scores, loss, train_step