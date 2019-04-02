import os
import tensorflow as tf
import numpy as np
import rejection_network


class RejectionSystem():

    def __init__(self):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self._train_dir = self.dir_path + "/Data/Train/"
        self._valid_dir = self.dir_path + "/Data/Valid/"
        # training setting
        self._training_epoches = 100
        self._number_of_minibatches = 20
        self._rejection_net = rejection_network.Network()
        self._initialize_training = True
        self._debug = False


    def load_data(self):
        train_images = np.load(self._train_dir + "train_images.npy")
        train_targets = np.load(self._train_dir + "train_targets.npy")
        valid_images = np.load(self._valid_dir + "valid_images.npy")
        valid_targets = np.load(self._valid_dir + "valid_targets.npy")
        return train_images, train_targets, valid_images, valid_targets

    def prepare_training_batches(self, inputs, targets):
        data_amount = np.shape(targets)[0]
        perm = np.arange(data_amount)
        np.random.shuffle(perm)
        inputs = inputs[perm]
        targets = targets[perm]
        inputs_batches = np.split(inputs, self._number_of_minibatches)
        targets_batches = np.split(targets, self._number_of_minibatches)
        return inputs_batches, targets_batches

    def train_model(self, train_images, train_targets, valid_images, valid_targets):
        TFgraph, images_placeholder, targets_placeholder, whether_training_placeholder, safety_scores, loss, train_step = self._rejection_net.build_rejection_network()
        model_loc = self._rejection_net.model_loc
        with TFgraph.as_default():
            with tf.Session() as sess:
                saver = tf.train.Saver()
                if(self._initialize_training):
                    print("Initialize parameters and train from scratch...")
                    sess.run(tf.global_variables_initializer())
                else:
                    print("Resume training on model loaded from {}...".format(model_loc))
                    saver.restore(sess, model_loc)
                for i in range(1, self._training_epoches+1):
                    train_images_batches, train_targets_batches = self.prepare_training_batches(train_images, train_targets)
                    train_loss_avg = 0
                    for j in range(self._number_of_minibatches):
                        _, train_loss, train_scores = sess.run([train_step, loss, safety_scores], feed_dict={
                            images_placeholder: train_images_batches[j],
                            targets_placeholder: train_targets_batches[j],
                            whether_training_placeholder: True
                        })
                        train_loss_avg += train_loss/self._number_of_minibatches
                    valid_loss, valid_scores = sess.run([loss, safety_scores], feed_dict={
                        images_placeholder: valid_images,
                        targets_placeholder: valid_targets,
                        whether_training_placeholder: False
                    })
                    if(self._debug):
                        print(valid_scores)
                    print("{}/{} Epoch. Avg CE: Train {} | Valid {}".format(i, self._training_epoches, train_loss_avg, valid_loss))
                    saver.save(sess, model_loc)
                    print("Trained model saved at {}!".format(model_loc))
        return

if(__name__=="__main__"):
    rejection_system = RejectionSystem()
    train_images, train_targets, valid_images, valid_targets = rejection_system.load_data()
    print("Data Loading Completed!")
    rejection_system.train_model(train_images, train_targets, valid_images, valid_targets)

