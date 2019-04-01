import numpy as np
import json

# TYPE = "TRAIN"
TYPE = "VALID"

if(__name__=="__main__"):
    if (TYPE == "TRAIN"):
        filename = "Data/Train/train_tags.json"
        filename_processed = "Data/Train/train_tags_processed.json"
        number_of_images = 5000
        target_filename = "Data/Train/train_targets.npy"
    else:
        filename = "Data/Valid/valid_tags.json"
        filename_processed = "Data/Valid/valid_tags_processed.json"
        number_of_images = 1000
        target_filename = "Data/Valid/valid_targets.npy"
    # first of all, make the json records contained within an array
    with open(filename, "r") as f_in:
        with open(filename_processed, "w") as f_out:
            for i, line in enumerate(f_in, 1):
                line = line.rstrip("\n")
                if(i==1): # first line, add start of array bracket
                    line = "["+line
                if(i==number_of_images): # last line, close the bracket for the array
                    line += "]"
                else: # add comma separating elements within the array
                    line += ","
                f_out.write(line+"\n")

    # use json to load the processed file as a list of records
    all_labels = np.zeros([number_of_images, 3])
    with open(filename_processed, "r") as f:
        all_records = json.load(f)
    assert np.size(all_records)==number_of_images
    for record in all_records:
        # obtain image index (starting from 0)
        token = record['content']
        image_index = int(token.split("_")[-2])
        assert image_index in range(number_of_images)
        assert np.sum(all_labels[image_index])==0 # ensure no repetitive labelling
        # obtain tagging targets
        tags = record['annotation']['labels']
        if('left' in tags or 'Left' in tags):
            all_labels[image_index][0] = 1
        if('straight' in tags or 'Straight' in tags):
            all_labels[image_index][1] = 1
        if('right' in tags or 'Right' in tags):
            all_labels[image_index][2] = 1
    print("# tagged: left: {}; straight: {}; right: {}".format(
        np.sum(all_labels[:,0]), np.sum(all_labels[:,1]), np.sum(all_labels[:,2])))
    np.save(target_filename, all_labels)
    print("Script finished successfully!")