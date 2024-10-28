"""
split all in one dataset to train/val/test

"""


import random
import pandas as pd
import numpy as np
import torch
import random


def use_fix_random_seed():
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)


if __name__ == "__main__":
    use_fix_random_seed()
    all_in_one_file = "dataset/all_in_one_with_subtitle_final.csv"
    data = pd.read_csv(all_in_one_file)
    vids = list(data["videoId"].values)

    random.shuffle(vids)

    total_vids = len(vids)
    train_num = round(total_vids * 0.7)
    validation_num = round(total_vids * 0.1)
    test_num = total_vids - train_num - validation_num

    train_vids = vids[:train_num]
    validation_vids = vids[train_num:(train_num + validation_num)]
    test_vids = vids[(train_num+validation_num):]

    print(f'train data: {len(train_vids)}')
    print(f'validation data: {len(validation_vids)}')
    print(f'test data: {len(test_vids)}')
    
    train_vid_file = "dataset/train_final.txt"
    with open(train_vid_file, "w") as f:
        for vid in train_vids:
            f.write(vid + "\n")
    
    validation_vid_file = "dataset/validation_final.txt"
    with open(validation_vid_file, "w") as f:
        for vid in validation_vids:
            f.write(vid + "\n")

    test_vid_file = "dataset/test_final.txt"
    with open(test_vid_file, "w") as f:
        for vid in test_vids:
            f.write(vid + "\n")



