"""
calculate various dataset statistics and visualization
"""


import os, glob
import shutil
import json
import pandas as pd
import numpy as np
from load_dataset_utils import parse_csv_to_list, extract_first_timestamp, clean_str


def draw_duration_hist():
    data = pd.read_csv("D:/py3_code/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv")
    vids = list(data["videoId"].values)

    durations = np.array(list(data["duration"].values))
    durations = durations[durations < 1800]
    duration_mean = np.mean(durations)
    print(len(durations))
    print(duration_mean)

    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(durations, bins=20)
    bins = list(bins)
    plt.show()
    print(bins)


def draw_chapter_num_hist():
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)

    chapter_nums = [len(x) for x in timestamps]
    print(f"min chapter num {min(chapter_nums)}")

    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(chapter_nums, bins=20)
    bins = list(bins)
    # plt.show()
    plt.savefig("/opt/tiger/video_chapter_youtube_dataset/chapter_num_hist.jpg")
    print(bins)


def timestamp_description_len():
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)

    description_list = []
    description_len_list = []
    for timestamp in timestamps:
        for line in timestamp:
            sec, description = extract_first_timestamp(line)
            description = clean_str(description)
            description_list.append(description)
            description_len_list.append(len(description.split(" ")))
    
    print(f"max description len {max(description_len_list)}")
    print(f"min description len {min(description_len_list)}")
    print(f"avg description len {sum(description_len_list)/len(description_len_list)}")



def count_all_images(vid_file):
    # data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    # all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)

    with open(vid_file, "r") as f:
        vids = f.readlines()
        vids = [x.strip() for x in vids]
    
    img_dir = "/opt/tiger/youtube_video_frame_dataset"
    image_num = 0
    for i, vid in enumerate(vids):
        # print(f"{i}/{len(vids)} {vid}...")
        all_images = glob.glob(img_dir + f"/{vid}/*.jpg")
        image_num += len(all_images)

    print(f"count all images {image_num}")


def count_all_clips(vid_file, clip_frame_num=16):
    img_dir = "/opt/tiger/youtube_video_frame_dataset"
    # processed vids
    with open(vid_file, "r") as f:
        vids = f.readlines()
        vids = [x.strip() for x in vids]

    all_clip_num = 0
    for vid in vids:
        # print(f"processing vid {vid}...")
        # image num
        image_path = os.path.join(img_dir, vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))

        # go through all clips within this video
        max_offset = 2
        clips = [[start_t, start_t + clip_frame_num] for start_t in range(0, image_num - clip_frame_num, 2 * max_offset)]
        all_clip_num += len(clips)
    
    print(f"count all clips {all_clip_num}")


def count_all_words(vid_file):
    with open(vid_file, "r") as f:
        vids = f.readlines()
        vids = [x.strip() for x in vids]
    
    asr_file_list = glob.glob("/opt/tiger/video_chapter_youtube_dataset/dataset/*/subtitle_*.json")
    asr_files = dict()
    for asr_file in asr_file_list:
        filename = os.path.basename(asr_file)
        vid = filename.split(".")[0][9:]
        asr_files[vid] = asr_file
    
    word_count_all = 0
    for vid in vids:
        asr_file = asr_files[vid]
        with open(asr_file, "r") as f:
            subtitles = json.load(f)
        
        text = ""
        for sub in subtitles:
            if len(text) == 0:
                text += sub["text"]
            else:
                text += " " + sub["text"]
        
        word_count = len(text.split(" "))
        word_count_all += word_count

    print(f"count all words {word_count_all}")


"""
calculate some statistics (chapter num, chapter title len, chapter duration) by categories
"""
def stats_by_category():
    with open("dataset/category2valid_vid.json", "r") as f:
        category2valid_vid = json.load(f)

    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)

    for category, vids in category2valid_vid.items():
        category_durations = []
        category_chapter_nums = []
        category_chapter_word_nums = []
        for vid in vids:
            idx = all_vids.index(vid)
            title = titles[idx]
            duration = durations[idx]
            timestamp = timestamps[idx]
            chapter_num = len(timestamp)
            chapter_word_num = 0
            for line in timestamp:
                sec, description = extract_first_timestamp(line)
                description = clean_str(description)
                chapter_word_num += len(description.split(" "))
            category_durations.append(duration)
            category_chapter_nums.append(chapter_num)
            category_chapter_word_nums.append(chapter_word_num)

        # print(f"{category}, avg duration {round(sum(category_durations) / len(category_durations), 2)}, \
        #     avg chapter duration {round(sum(category_durations) / sum(category_chapter_nums), 2)}, \
        #     avg chapter num {round(sum(category_chapter_nums)/ len(category_chapter_nums), 2)}, \
        #         avg chapter word num {round(sum(category_chapter_word_nums)/sum(category_chapter_nums), 2)}")

        print(f"{category} & {len(vids)} & {round(sum(category_durations) / sum(category_chapter_nums), 2)} & {round(sum(category_chapter_nums)/ len(vids), 2)} & {round(sum(category_chapter_word_nums)/sum(category_chapter_nums), 2)}\\\\")


if __name__ == "__main__":
    # timestamp_description_len()

    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/new_train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/new_test.txt"
    validation_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/new_validation.txt"

    count_all_images(train_vid_file)
    count_all_images(validation_vid_file)
    count_all_images(test_vid_file)

    count_all_clips(train_vid_file)
    count_all_clips(validation_vid_file)
    count_all_clips(test_vid_file)

    count_all_words(train_vid_file)
    count_all_words(validation_vid_file)
    count_all_words(test_vid_file)

    # stats_by_category()

