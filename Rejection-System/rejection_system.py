from __future__ import print_function

import os

import scipy

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

import rejection_network


class RejectionSystem():

    def __init__(self):
        import os
        self.dir_path = os.path.dirname(__file__)
        self._model_path = self.dir_path + '/rejection_model/'
        self._train_dir = self.dir_path + "Data/Train/"
        self._valid_dir = self.dir_path + "Data/Valid/"
        self._amount_of_commands = 4 # [follow lane, left, right, go straight]
        # training setting
        self._training_epoches = 50
        self._minibatch_amount = 100
        self._rejection_net = rejection_network.Network()


    def load_data(self):
        train_images = np.load(self._train_dir + "images.npy")
        train_commands = np.load(self._train_dir + "commands.npy")
        valid_images = np.load(self._valid_dir + "images.npy")
        valid_commands = np.load(self._valid_dir + "commands.npy")
        # Convert targets into one-hot format
        train_amount = np.size(train_commands); assert np.shape(train_commands)==(train_amount, )
        valid_amount = np.size(valid_commands); assert np.shape(valid_commands)==(valid_amount, )
        train_commands_onehot = np.zeros([train_amount, self._amount_of_commands])
        # Follow_lane: element 0; Left: element 1; Right: element 2; Straight: element 3
        train_commands_onehot[np.arange(train_amount), train_commands.astype(int)-2]=1
        valid_commands_onehot = np.zeros([valid_amount, self._amount_of_commands])
        valid_commands_onehot[np.arange(valid_amount), valid_commands.astype(int)-2] = 1
        return train_images, train_commands_onehot, valid_images, valid_commands_onehot

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
        TFgraph, images_placeholder, targets_placeholder, safety_scores, loss, train_step = self._rejection_net.build_rejection_network()
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
                    if(i%1==0):
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
    rejection_system = RejectionSystem()
    train_images, train_targets, valid_images, valid_targets = rejection_system.load_data()
    print("Data Loading Completed!")
    rejection_system.train_model(train_images, train_targets, valid_images, valid_targets)
