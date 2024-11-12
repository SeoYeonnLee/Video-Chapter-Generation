# calculate various dataset statistics and visualization

import os, glob
import shutil
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list, extract_first_timestamp, clean_str

def draw_duration_hist(data_file, save_path):
    data = pd.read_csv(data_file)
    vids = list(data["videoId"].values)

    durations = np.array(list(data["duration"].values))
    durations = durations[durations < 1800]
    duration_mean = round(np.mean(durations), 2)
    print(f'Number of Videos : {len(durations):,}')
    print(f'Mean Video duration : {duration_mean}')

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(durations, bins=20)
    plt.title('Video Duration Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of Videos')
    bins = list(bins)
    plt.savefig(os.path.join(save_path, "hist/mean_duration_hist.jpg"))
    plt.close()

    return len(durations), duration_mean

def draw_chapter_num_hist(data_file, save_path):
    all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)

    chapter_nums = [len(x) for x in timestamps]
    print(f"Min Chapter Num : {min(chapter_nums):,}")
    print(f"Max Chapter Num : {max(chapter_nums):,}")
    print(f"Avg Chapter Num : {round(sum(chapter_nums)/len(chapter_nums), 2)}")

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(chapter_nums, bins=20)
    plt.title('Video Chapter Distribution')
    plt.xlabel('Number of Chapters')
    plt.ylabel('Number of Videos')
    bins = list(bins)
    plt.savefig(os.path.join(save_path, "hist/chapter_num_hist.jpg"))
    plt.close() 
    
    return min(chapter_nums), max(chapter_nums), round(sum(chapter_nums)/len(chapter_nums), 2)

def timestamp_description_len(data_file, save_path):
    all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)

    description_list = []
    description_len_list = []
    for timestamp in timestamps:
        for line in timestamp:
            sec, description = extract_first_timestamp(line)
            description = clean_str(description)
            description_list.append(description)
            description_len_list.append(len(description.split(" ")))
    
    print(f"Max description len : {max(description_len_list):,}")
    print(f"Min description len : {min(description_len_list):,}")
    print(f"Avg description len : {round(sum(description_len_list)/len(description_len_list), 2)}")

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(description_len_list, bins=20)
    plt.title('Description Length Distribution')
    plt.xlabel('Description Length')
    plt.ylabel('Number of Videos')
    bins = list(bins)
    plt.savefig(os.path.join(save_path, "hist/description_len_hist.jpg"))
    plt.close() 

    return max(description_len_list), min(description_len_list), round(sum(description_len_list)/len(description_len_list), 2)

def count_all_images(vid_file, img_dir):
    with open(vid_file, "r") as f:
        vids = f.readlines()
        vids = [x.strip() for x in vids]
    
    image_num = 0
    for vid in tqdm(vids, desc=f"Counting Images from {os.path.basename(vid_file)}", unit="video"):
        all_images = glob.glob(img_dir + f"/{vid}/*.jpg")
        image_num += len(all_images)

    print(f"Count All Images : {image_num:,}")

    return image_num

def count_all_clips(vid_file, img_dir, clip_frame_num=16):
    # processed vids
    with open(vid_file, "r") as f:
        vids = f.readlines()
        vids = [x.strip() for x in vids]

    all_clip_num = 0
    for vid in tqdm(vids, desc=f"Counting Clips from {os.path.basename(vid_file)}", unit="video"):
        # image num
        image_path = os.path.join(img_dir, vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))

        # go through all clips within this video
        max_offset = 2
        clips = [[start_t, start_t + clip_frame_num] for start_t in range(0, image_num - clip_frame_num, 2 * max_offset)]
        all_clip_num += len(clips)
    
    print(f"Count All Clips : {all_clip_num:,}")

    return all_clip_num

def count_all_words(vid_file):
    with open(vid_file, "r") as f:
        vids = f.readlines()
        vids = [x.strip() for x in vids]
    
    asr_file_list = glob.glob("dataset/*/subtitle_*.json")
    asr_files = dict()
    for asr_file in asr_file_list:
        filename = os.path.basename(asr_file)
        vid = filename.split(".")[0][9:]
        asr_files[vid] = asr_file
    
    word_count_all = 0
    missing_subtitle_vids = []

    for vid in tqdm(vids, desc=f"Counting Words from {os.path.basename(vid_file)}", unit="video"):
        if vid not in asr_files:
            missing_subtitle_vids.append(vid)
            continue
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

    print(f"Count All Words : {word_count_all:,}")
    print(f"Number of Videos Missing Subtitles : {len(missing_subtitle_vids):,}")

    return word_count_all, missing_subtitle_vids

# calculate some statistics (chapter num, chapter title len, chapter duration) by categories

def stats_by_category(data_file, category_file):
    with open(category_file, "r") as f:
        category2valid_vid = json.load(f)

    all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)
    
    category_stats = {}

    for category in tqdm(category2valid_vid.keys(), desc="Processing Categories", unit="category"):
        vids = category2valid_vid[category]
        category_durations = []
        category_chapter_nums = []
        category_chapter_word_nums = []
        chapter_durations = []

        for vid in tqdm(vids, desc=f"Processing Videos in {category}", unit="video", leave=False):
            idx = all_vids.index(vid)

            title = titles[idx]
            duration = durations[idx]
            timestamp = timestamps[idx]

            chapter_num = len(timestamp)

            sorted_timestamps = []
            for line in timestamp:
                sec, description = extract_first_timestamp(line)
                sorted_timestamps.append(sec)
            
            sorted_timestamps.sort()  # 시간순 정렬
            
            # 각 챕터의 duration 계산
            for i in range(len(sorted_timestamps)):
                if i == len(sorted_timestamps) - 1:
                    # 마지막 챕터는 비디오 끝까지
                    chapter_dur = duration - sorted_timestamps[i]
                else:
                    # 다음 챕터 시작까지
                    chapter_dur = sorted_timestamps[i+1] - sorted_timestamps[i]
                chapter_durations.append(chapter_dur)

            chapter_word_num = 0
            for line in timestamp:
                sec, description = extract_first_timestamp(line)
                description = clean_str(description)
                chapter_word_num += len(description.split(" "))

            category_durations.append(duration)
            category_chapter_nums.append(chapter_num)
            category_chapter_word_nums.append(chapter_word_num)

        stats = {
            "video_count": len(vids),
            "total_duration": sum(category_durations),
            "total_chapters": sum(category_chapter_nums),
            "total_words": sum(category_chapter_word_nums),
            "avg_chapter_duration": round(sum(category_durations) / sum(category_chapter_nums), 2),
            "avg_chapters_per_video": round(sum(category_chapter_nums) / len(vids), 2),
            "avg_words_per_chapter": round(sum(category_chapter_word_nums) / sum(category_chapter_nums), 2),
            "avg_video_duration": round(sum(category_durations) / len(vids), 2),
            "avg_chapter_duration": round(sum(chapter_durations) / len(chapter_durations), 2),
            "min_chapter_duration": round(min(chapter_durations), 2),
            "max_chapter_duration": round(max(chapter_durations), 2)
        }
        
        category_stats[category] = stats
    
    total_stats = {
        "total_videos": sum(stats['video_count'] for stats in category_stats.values()),
        "total_chapters": sum(stats['total_chapters'] for stats in category_stats.values()),
        "total_words": sum(stats['total_words'] for stats in category_stats.values()),
        "avg_video_duration": round(sum(stats['total_duration'] for stats in category_stats.values()) / sum(stats['video_count'] for stats in category_stats.values()), 2),
        "avg_chapters_per_video": round(sum(stats['total_chapters'] for stats in category_stats.values()) / sum(stats['video_count'] for stats in category_stats.values()), 2),
        "avg_words_per_chapter": round(sum(stats['total_words'] for stats in category_stats.values()) / sum(stats['total_chapters'] for stats in category_stats.values()), 2)
    }

    return category_stats, total_stats

def main():
    train_vid_file = "dataset/final_train.txt"
    valid_vid_file = "dataset/final_validation.txt"
    test_vid_file = "dataset/final_test.txt"

    data_file="dataset/all_in_one_with_subtitle_final.csv"
    category_file="dataset/sampled_videos.json"

    img_dir="youtube_video_frame_dataset"
    save_path="dataset_stats_result"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 기본 데이터 분포 확인 및 저장
    video_nums, duration_mean = draw_duration_hist(data_file, save_path)
    min_chapter_nums, max_chapter_nums, mean_chapter_nums = draw_chapter_num_hist(data_file, save_path)
    max_description_len, min_description_len, avg_description_len = timestamp_description_len(data_file, save_path)

    # 실제 데이터 양 계산
    print('***** Start count all images *****')
    train_img_num = count_all_images(train_vid_file, img_dir)
    valid_img_num = count_all_images(valid_vid_file, img_dir)
    test_img_num = count_all_images(test_vid_file, img_dir)

    print('***** Start count all clips *****')
    train_clip_num = count_all_clips(train_vid_file, img_dir)
    valid_clip_num = count_all_clips(valid_vid_file, img_dir)
    test_clip_num = count_all_clips(test_vid_file, img_dir)

    print('***** Start count all words *****')
    train_word_num, train_missing_subs  = count_all_words(train_vid_file)
    valid_word_num, valid_missing_subs  = count_all_words(valid_vid_file)
    test_word_num, test_missing_subs  = count_all_words(test_vid_file)

    # 카테고리별 메타데이터 분석
    category_stats, total_stats = stats_by_category(data_file, category_file)

    # 데이터 저장
    base_stats  ={}
    stats = {
        "total_video_nums" : video_nums,
        "duration_mean" : duration_mean,
        "min_chapter_nums" : min_chapter_nums,
        "max_chapter_nums" : max_chapter_nums,
        "mean_chapter_nums" : mean_chapter_nums,
        "max_description_len" : max_description_len,
        "min_description_len" : min_description_len,
        "avg_description_len" : avg_description_len
    }

    train_nums = {
        "train_img_num" : train_img_num,
        "train_clip_num" : train_clip_num,
        "train_word_num" : train_word_num,
        "train_missing_subtitle_videos" : train_missing_subs,
        "train_missing_subtitle_count": len(train_missing_subs)
    }

    valid_nums = {
        "valid_img_num" : valid_img_num,
        "valid_clip_num" : valid_clip_num,
        "valid_word_num" : valid_word_num,
        "valid_missing_subtitle_videos": valid_missing_subs,
        "valid_missing_subtitle_count": len(valid_missing_subs)
    }

    test_nums = {
        "test_img_num" : test_img_num,
        "test_clip_num" : test_clip_num,
        "test_word_num" : test_word_num,
        "test_missing_subtitle_videos": test_missing_subs,
        "test_missing_subtitle_count": len(test_missing_subs)
    }
    
    base_stats['stats'] = stats
    base_stats['train_nums'] = train_nums
    base_stats['valid_nums'] = valid_nums
    base_stats['test_nums'] = test_nums

    with open(os.path.join(save_path, "json/base_statistics.json"), "w") as f:
        json.dump(base_stats, f, indent=4)

    with open(os.path.join(save_path, "json/category_statistics.json"), "w") as f:
        json.dump(category_stats, f, indent=4)

    with open(os.path.join(save_path, "json/total_statistics.json"), "w") as f:
        json.dump(total_stats, f, indent=4)


if __name__ == "__main__":
    main()