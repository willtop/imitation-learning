# Parse data and store in numpy array form for both training and validation data

import numpy as np
import os
import h5py
from local_settings import *

# Each file has 200 data points
timeslot_interleaving = 10 # Take 20 data points from the 200 data points. One per 4 time frames
Train_h5_file_amount = 500
Valid_h5_file_amount = 50

def load_data_per_file(file_name):
    data = h5py.File(file_name, "r")
    image = np.array(data['rgb'])
    assert np.shape(image) == (200, 88, 200, 3)
    # Crop the input images
    image = image[:,9:79,20:180,:] # Dimension: 88X200->70X120
    # time slots taken with interleaving
    time_slots_taken = np.arange(0, 200, timeslot_interleaving)
    image = image[time_slots_taken,:,:,:]
    return image

if(__name__=="__main__"):
    original_train_dir = "../Original_Data/SeqTrain/"
    original_valid_dir = "../Original_Data/SeqVal/"
    target_train_dir = "Data/Train/"
    target_valid_dir = "Data/Valid/"
    # Load training data
    images = []
    for i in range(Train_h5_file_amount):
        file_name = original_train_dir + "data_{0:05}.h5".format(3663+i)
        image = load_data_per_file(file_name)
        images.append(image)
        if((i+1)%50==0):
            print("Finished loading {}/{} files".format(i+1, Train_h5_file_amount))
    images = np.concatenate(images, axis=0)
    assert np.shape(images) == (int(Train_h5_file_amount*200/timeslot_interleaving), IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    np.save(target_train_dir+"images.npy", images)
    # Load validation data
    images = []
    targets = []
    for i in range(Valid_h5_file_amount):
        file_name = original_valid_dir+"data_{0:05}.h5".format(i)
        image = load_data_per_file(file_name)
        images.append(image)
    images = np.concatenate(images, axis=0)
    assert np.shape(images) == (int(Valid_h5_file_amount*200/timeslot_interleaving), IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    np.save(target_valid_dir + "images.npy", images)

    print("Data Parsing Successful!")


