import numpy as np
from scipy.misc import imsave
import os
from local_settings import *

# TYPE = "TRAIN"
TYPE = "VALID"

if(__name__=="__main__"):
    if(TYPE=="TRAIN"):
        data_dir = "Data/Train/"
    else:
        data_dir = "Data/Valid/"
    print("loading image npy file from {}...".format(data_dir))
    images = np.load(data_dir+"train_images.npy")
    images_amount = np.shape(images)[0]
    for i in range(images_amount):
        image = images[i]
        assert np.shape(image)==(IMAGE_HEIGHT,IMAGE_WIDTH,3)
        jpg_pic_name = "{}_{}.png".format(i, TYPE)
        imsave(data_dir+"jpg_images/"+jpg_pic_name, image)
        if(((i+1)*100/images_amount)%10==0):
            print("Finished {}/{} images...".format(i, images_amount))
    print("Convertion completed!")
