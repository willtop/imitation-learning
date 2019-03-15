from __future__ import print_function

import os

import scipy

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

from rejection_network import load_rejection_network


class RejectionSystem():

    def __init__(self, avoid_stopping, memory_fraction=0.25, image_cut=[115, 510]):

        import os
        self.dir_path = os.path.dirname(__file__)
        self._model_path = self.dir_path + '/rejection_model/'
        self._training_epoches = 50
        self._train_dir = self.dir_path + "Data/Train/"
        self._valid_dir = self.dir_path + "Data/Valid/"
        self._minibatch_amount = 100

    def load_data(self):
        train_images = np.load(self._train_dir + "images.npy")
        train_commands = np.load(self._train_dir + "commands.npy")
        valid_images = np.load(self._valid_dir + "images.npy")
        valid_commands = np.load(self._valid_dir + "commands.npy")
        return train_images, train_commands, valid_images, valid_commands

    def prepare_training_batches(self, inputs, targets):
        data_amount = np.shape(targets)[0]
        perm = np.arange(data_amount)
        np.random.shuffle(perm)
        inputs = inputs[perm]
        targets = targets[perm]
        inputs_batches = np.split(inputs, self._minibatch_amount)
        targets_batches = np.split(targets, self._minibatch_amount)
        return inputs_batches, targets_batches

    def train_model(self, train_images, train_commands, valid_images, valid_commands):
        TFgraph, images_placeholder, targets_placeholder, safety_scores, loss, train_step = load_rejection_network()
        with TFgraph.as_default():
            with tf.Session() as sess:
                saver = tf.train.saver()
                sess.run(tf.global_variables_initializer())
                for i in range(1, self._training_epoches+1):
                    train_images_batches, train_commands_batches = self.prepare_training_batches(train_images,
                                                                                                 train_commands)
                    for j in range(self._minibatch_amount):
                        _, train_loss = sess.run([train_step, loss], feed_dict={
                            images_placeholder: train_images_batches[j],
                            targets_placeholder: train_commands_batches[j]
                        })
                    if(i%10==0):
                        valid_loss = sess.run(loss, feed_dict={
                            images_placeholder: valid_images,
                            targets_placeholder: valid_commands
                        })
                        print("{}/{} Epoch. Avg Weighted CE: Train {} | Valid {}".format(
                            i, self._training_epoches, train_loss, valid_loss))
                saver.save(sess, self._model_path)
                print("Trained model saved at {}!".format(self._model_path))
        return

if(__name__=="__main__"):
    rejection_system =  RejectionSystem()
    train_images, train_targets, valid_images, valid_targets = rejection_system.load_data()
    rejection_system.train_model(train_images, train_targets, valid_images, valid_targets)

