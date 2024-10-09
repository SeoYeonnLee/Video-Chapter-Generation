"""
split dataset to easy and hard

"""

import math
import pandas as pd


easy_hard_labeling_data = "./dataset/easy_hard_labeling.csv"
data = pd.read_csv(easy_hard_labeling_data)

vids = list(data["object id"].values)
results1 = list(data["1_label_result"].values)
results2 = list(data["2_label_result"].values)


vid2result = dict()
easy_vid = []
hard_vid = []
ambiguous_vid = []
wrong_data_vid = []
for vid, result1, result2 in zip(vids, results1, results2):
    if math.isnan(result2):
        vid2result[vid] = result1
        label = result1
    else:
        vid2result[vid] = result2
        label = result2

    if label == 2:
        easy_vid.append(vid)
    if label == 1:
        hard_vid.append(vid)
    if label == 0:
        ambiguous_vid.append(vid)
    if label == -1:
        wrong_data_vid.append(vid)


with open("./dataset/easy_vid.txt", "w") as f:
    for vid in easy_vid:
        f.write(f"{vid}\n")
with open("./dataset/hard_vid.txt", "w") as f:
    for vid in hard_vid:
        f.write(f"{vid}\n")
with open("./dataset/ambiguous_vid.txt", "w") as f:
    for vid in ambiguous_vid:
        f.write(f"{vid}\n")
with open("./dataset/wrong_data_vid.txt", "w") as f:
    for vid in wrong_data_vid:
        f.write(f"{vid}\n")



