import os
import tensorflow as tf
import numpy as np
import rejection_network


def load_data_for_inference(self):
    valid_images = np.load(os.path.dirname(os.path.abspath(__file__)) + "/Data/Valid/valid_images.npy")
    valid_targets = np.load(os.path.abspath(__file__)) + "/Data/Valid/valid_targets.npy")
    return valid_images, valid_targets


def model_inference(valid_images, valid_targets):
    rejection_net = rejection_network.Network()
    model_loc = rejection_net.model_loc
    TFgraph, images_placeholder, targets_placeholder, whether_training_placeholder, safety_scores, loss, train_step = rejection_net.build_rejection_network()
    with TFgraph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver()
            print("Inference on model loaded from {}...".format(model_loc))
            saver.restore(sess, model_loc)
            safety_scores = sess.run(safety_scores, feed_dict={
                images_placeholder: valid_images,
                targets_placeholder: valid_targets,
                whether_training_placeholder: False
            })
    # Convert to binary acceptable commands
    acceptable_commands = (safety_scores>=0.5).astype(int)
    assert np.shape(acceptable_commands)==(np.shape(valid_images)[0], rejection_net.number_of_commands)
    return acceptable_commands

if(__name__=="__main__"):
    valid_images, valid_targets = load_data_for_inference()
    print("Data Loading Completed!")
    valid_commands = model_inference(valid_images, valid_targets)
    
    print("Script finished successfully!")