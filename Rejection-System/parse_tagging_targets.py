TYPE = "TRAIN"
# TYPE = "VALID"

if(__name__=="__main__"):
    if (TYPE == "TRAIN"):
        filename = "Data/Train/train_tags.json"
    else:
        filename = "Data/Valid/valid_tags.json"
    with open(filename, "r") as f:
        for line in f:
            tokens = line.split(":")
            img_index = int(((tokens[2]).split(",")[0]).split("_")[-2])
            assert 0<=img_index<5000
            labels = tokens[4].split("note")[0][2:-4]
            if(label=='straight'):
                label = [0,1,0]
