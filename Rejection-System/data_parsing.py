# Parse data and store in numpy array form for both training and validation data

import numpy as np
import os
import h5py

# Each file has 200 data points
timeslot_interleaving = 4 # Take 50 data points from the 200 data points. One per 4 time frames
Train_h5_file_amount = 2000
Valid_h5_file_amount = 200

def load_data_per_file(file_name):
    data = h5py.File(file_name, "r")
    image = np.array(data['rgb'])
    assert np.shape(image) == (200, 88, 200, 3)
    target = np.array(data['targets'])
    assert np.shape(target) == (200, 28)
    # Crop the input images
    image = image[:,9:79,40:160,:] # Dimension: 88X200->70X120
    # only taking the directional command:
    target = target[:, 24]
    # time slots taken with interleaving
    time_slots_taken = np.arange(0, 200, timeslot_interleaving)
    image = image[time_slots_taken,:,:,:]
    target = target[time_slots_taken]
    return image, target

if(__name__=="__main__"):
    current_dir = os.path.dirname(__file__)
    original_train_dir = current_dir + "../Original_Data/SeqTrain/"
    original_valid_dir = current_dir + "../Original_Data/SeqVal/"
    target_train_dir = current_dir + "Data/Train/"
    target_valid_dir = current_dir + "Data/Valid/"
    # Load training data
    images = []
    targets = []
    for i in range(Train_h5_file_amount):
        file_name = original_train_dir + "data_{0:05}.h5".format(3663+i)
        image, target = load_data_per_file(file_name)
        images.append(image)
        targets.append(target)
        if((i+1)%50==0):
            print("Finished loading {}/{} files".format(i+1, Train_h5_file_amount))
    images = np.concatenate(images, axis=0)
    assert np.shape(images) == (int(Train_h5_file_amount*200/timeslot_interleaving), 70, 120, 3)
    targets = np.concatenate(targets)
    assert np.shape(targets) == (int(Train_h5_file_amount*200/timeslot_interleaving),)
    np.save(target_train_dir+"images.npy", images)
    np.save(target_train_dir+"commands.npy", targets)
    # Load validation data
    images = []
    targets = []
    for i in range(Valid_h5_file_amount):
        file_name = original_valid_dir+"data_{0:05}.h5".format(i)
        image, target = load_data_per_file(file_name)
        images.append(image)
        targets.append(target)
    images = np.concatenate(images, axis=0)
    assert np.shape(images) == (int(Valid_h5_file_amount*200/timeslot_interleaving), 70, 120, 3)
    targets = np.concatenate(targets)
    assert np.shape(targets) == (int(Valid_h5_file_amount*200/timeslot_interleaving),)
    np.save(target_valid_dir + "images.npy", images)
    np.save(target_valid_dir + "commands.npy", targets)

    print("Data Parsing Successful!")


