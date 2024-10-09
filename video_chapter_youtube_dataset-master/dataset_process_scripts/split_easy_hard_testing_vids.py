
test_vid_file = "dataset/new_test.txt"
easy_vid_file = "dataset/easy_vid.txt"
hard_vid_file = "dataset/hard_vid.txt"

easy_test_vid_file = "dataset/new_easy_test_vid.txt"
hard_test_vid_file = "dataset/new_hard_test_vid.txt"


with open(test_vid_file, "r") as f:
    test_vids = [x.strip() for x in f.readlines()]

with open(easy_vid_file, "r") as f:
    easy_vids = [x.strip() for x in f.readlines()]

with open(hard_vid_file, "r") as f:
    hard_vids = [x.strip() for x in f.readlines()]


easy_test_vids = []
hard_test_vids = []
for test_vid in test_vids:
    if test_vid in easy_vids:
        easy_test_vids.append(test_vid)
    if test_vid in hard_vids:
        hard_test_vids.append(test_vid)

with open(easy_test_vid_file, "w") as f:
    for vid in easy_test_vids:
        f.write(f"{vid}\n")
with open(hard_test_vid_file, "w") as f:
    for vid in hard_test_vids:
        f.write(f"{vid}\n")  

