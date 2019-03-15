# Parse data and store in numpy array form for both training and validation data

import numpy as np
import os
import h5py

# Each file has 200 data points
Train_h5_file_amount = 500
Valid_h5_file_amount = 50

def load_data_per_file(file_name):
    data = h5py.File(file_name, "r")
    image = np.array(data['rgb'])
    assert np.shape(image) == (200, 88, 200, 3)
    target = np.array(data['targets'])
    assert np.shape(target) == (200, 28)
    # only taking the directional command:
    target = target[:, 24]
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
    assert np.shape(images) == (Train_h5_file_amount*200, 88, 200, 3)
    targets = np.concatenate(targets)
    assert np.shape(targets) == (Train_h5_file_amount*200,)
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
    assert np.shape(images) == (Valid_h5_file_amount * 200, 88, 200, 3)
    targets = np.concatenate(targets)
    assert np.shape(targets) == (Valid_h5_file_amount * 200,)
    np.save(target_valid_dir + "images.npy", images)
    np.save(target_valid_dir + "commands.npy", targets)

    print("Data Parsing Successful!")


