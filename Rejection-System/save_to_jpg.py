import numpy as np
from scipy.misc import imsave
import os
import local_settings

# TYPE = "TRAIN"
TYPE = "VALID"

if(__name__=="__main__"):
    main_dir = os.path.dirname(os.path.realpath(__file__))
    if(TYPE=="TRAIN"):
        data_dir = main_dir + "/Data/Train/"
    else:
        data_dir = main_dir + "/Data/Valid/"
    print("loading image npy file from {}...".format(data_dir))
    images = np.load(data_dir+"images.npy")
    images_amount = np.shape(images)[0]
    for i in range(images_amount):
        image = images[i]
        assert np.shape(image)==(IMAGE_HEIGHT,IMAGE_WIDTH,3)
        jpg_pic_name = "{}_{}.png".format(i, TYPE)
        imsave(data_dir+"jpg_images/"+jpg_pic_name, image)
        if(((i+1)*100/images_amount)%10==0):
            print("Finished {}/{} images...".format(i, images_amount))
    print("Convertion completed!")
