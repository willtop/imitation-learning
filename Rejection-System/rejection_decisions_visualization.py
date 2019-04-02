import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import rejection_network
from local_settings import *

MAKE_VIDEO = True
Visualization_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Test_Results/")

def visualize_commands(images, commands):
    number_of_images = np.shape(images)[0]
    for i in range(number_of_images):
        plt.figure(figsize=[VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT])
        img = images[i]
        command = commands[i]
        assert np.shape(command)==(3,)
        plt.imshow(img)
        if(command[0]==1): # can turn left
            plt.arrow(IMAGE_WIDTH/2-10,IMAGE_HEIGHT-10,-10,0,width=1,color='green')
        if(command[1]==1): # can go straight
            plt.arrow(IMAGE_WIDTH/2,IMAGE_HEIGHT-10,0,-10,width=1,color='green')
        if(command[2]==1): # can turn right
            plt.arrow(IMAGE_WIDTH/2+10,IMAGE_HEIGHT-10,10,0,width=1,color='green')
        plt.pause(0.2)
        plt.savefig(Visualization_Dir+"{}.png".format(i))
        plt.close()
        if((i+1)%10==0):
            print("Showed {}/{} images".format(i+1, number_of_images))
    return

def load_data_for_inference():
    valid_images = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Valid/valid_images.npy"))
    valid_targets = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Valid/valid_targets.npy"))
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
    print("Obtained rejection network's inference results!")
    return acceptable_commands

if(__name__=="__main__"):
    if(MAKE_VIDEO):
        print("Making videos based on images from {}...".format(Visualization_Dir))
        import cv2
        image_filenames = [filename for filename in os.listdir(Visualization_Dir) if filename.endswith('png')]
        # Get the common image dimension
        sample_img = cv2.imread(os.path.join(Visualization_Dir, image_filenames[10]))
        frame_height, frame_width, frame_channels = sample_img.shape
        # Create an avi video
        video_output = cv2.VideoWriter(os.path.join(Visualization_Dir, "video_output.avi"), 0, 1, (frame_width, frame_height))
        for image_filename in image_filenames:
            video_output.write(cv2.imread(os.path.join(Visualization_Dir, image_filename)))
        video_output.release()
        cv2.destroyAllWindows()
        print("Video making successful!")
    else:
        print("Getting rejection network inference results and obtain images with arrows...")
        valid_images, valid_targets = load_data_for_inference()
        print("Data Loading Completed!")
        valid_commands = model_inference(valid_images, valid_targets)
        visualize_commands(valid_images, valid_commands)
    print("Script finished successfully!")
