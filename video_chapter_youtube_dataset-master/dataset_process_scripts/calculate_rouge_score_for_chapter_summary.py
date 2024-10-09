"""
This script calculates rouge score to see how much overlap between original subtitle and chapter description

"""

import sys
import glob, os
import json
from rouge import Rouge 
from dataset_process_scripts.load_dataset_utils import load_dataset_with_subtitle, extract_first_timestamp, parse_csv_to_list


# set a large stack size since the rouge-L calculation is recursive
# print(sys.getrecursionlimit())
sys.setrecursionlimit(8735 * 2080 + 10)


def calculate_rouge_score_for_video(all_vids, vid2title, vid2timestamps, vid2durations, vid2asr_files):
    rouge = Rouge()

    inconsistent_chapter_length_vids = []
    score_calculate_error_vids = []
    rouge1_r = []
    rouge2_r = []
    rougel_r = []

    token512_rouge1_r = []
    token512_rouge2_r = []
    token512_rougel_r = []

    chapter_text_length = []
    for i, vid in enumerate(all_vids):
        # vid = "0KAPUlqKhEw"
        timestamp = vid2timestamps[vid]
        title = vid2title[vid]
        duration = vid2durations[vid]
        asr_file = vid2asr_files[vid]

        view_link = f"https://www.youtube.com/watch?v={vid}"
        print(f"{i}/{len(all_vids)}, {view_link}   {title}")

        with open(asr_file, "r") as f:
            subtitle = json.load(f)

        # extract timestamp
        timepoint_secs = []
        descriptions = []
        for line in timestamp:
            sec, description = extract_first_timestamp(line)
            timepoint_secs.append(sec)
            descriptions.append(description)

        # get all subtitle within a chapter
        timepoint_idx = 0
        time_gap = 2
        text_within_chapter = dict()
        for sub in subtitle:
            text = sub["text"]
            start = sub["start"]

            if timepoint_idx < len(timepoint_secs) and start > timepoint_secs[timepoint_idx]:
                timepoint_idx += 1

            chapter_start_t = timepoint_secs[timepoint_idx - 1] - time_gap
            if timepoint_idx < len(timepoint_secs):
                chapter_end_t = timepoint_secs[timepoint_idx] + time_gap
            else:
                chapter_end_t = duration + time_gap

            if chapter_start_t < start < chapter_end_t:
                key = timepoint_idx - 1
                if key in text_within_chapter:
                    text_within_chapter[key] += " " + text
                else:
                    text_within_chapter[key] = text
        
        text_within_chapter = list(text_within_chapter.values())
        if len(text_within_chapter) != len(descriptions):
            inconsistent_chapter_length_vids.append(vid)
            continue
        
        
        # stats avg chapter text length
        token512_text_within_chapter = []
        for chapter_text in text_within_chapter:
            token_list = chapter_text.split(" ")
            chapter_text_length.append(len(token_list))

            if len(token_list) > 512:
                print()
            token512_list = token_list[:512]
            token512_text = " ".join(token512_list)
            token512_text_within_chapter.append(token512_text)

        # calculate rouge score
        try:
            scores = rouge.get_scores(token512_text_within_chapter, descriptions, avg=True)
            token512_rouge1_r.append(scores["rouge-1"]["r"])
            token512_rouge2_r.append(scores["rouge-2"]["r"])
            token512_rougel_r.append(scores["rouge-l"]["r"])

            scores = rouge.get_scores(text_within_chapter, descriptions, avg=True)
            rouge1_r.append(scores["rouge-1"]["r"])
            rouge2_r.append(scores["rouge-2"]["r"])
            rougel_r.append(scores["rouge-l"]["r"])
            print(scores)
        except Exception as e:
            score_calculate_error_vids.append(vid)
            print("error happen in rouge score calculation, skip this video")

        # scores = rouge.get_scores(text_within_chapter, descriptions)
        # rouge1_recall = []
        # rouge2_recall = []
        # rougel_recall = []
        # for score in scores:
        #     rouge1_recall.append(score["rouge-1"]["r"])
        #     rouge2_recall.append(score["rouge-2"]["r"])
        #     rougel_recall.append(score["rouge-l"]["r"])
        # print(rouge1_recall)
        # print(rouge2_recall)
        # print(rougel_recall)

    print("inconsistent_chapter_length_vids:")
    print(inconsistent_chapter_length_vids)
    print("score_calculate_error_vids:")
    print(score_calculate_error_vids)

    return rouge1_r, rouge2_r, rougel_r, token512_rouge1_r, token512_rouge2_r, token512_rougel_r, chapter_text_length





if __name__ == "__main__":
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"

    all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)
    vid2title = dict()
    vid2timestamps = dict()
    vid2durations = dict()
    for i in range(len(all_vids)):
        vid = all_vids[i]
        vid2title[vid] = titles[i]
        vid2timestamps[vid] = timestamps[i]
        vid2durations[vid] = durations[i]
    
    asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
    vid2asr_files = dict()
    for asr_file in asr_file_list:
        filename = os.path.basename(asr_file)
        vid = filename.split(".")[0][9:]
        vid2asr_files[vid] = asr_file

    # all_vids = all_vids[:100]
    rouge1_r, rouge2_r, rougel_r, token512_rouge1_r, token512_rouge2_r, token512_rougel_r, chapter_text_length = calculate_rouge_score_for_video(all_vids, vid2title, vid2timestamps, vid2durations, vid2asr_files)
    


    # visualize histogram
    import matplotlib.pyplot as plt
    import numpy as np
    rouge1_r = np.array(rouge1_r)
    rouge2_r = np.array(rouge2_r)
    rougel_r = np.array(rougel_r)
    token512_rouge1_r = np.array(token512_rouge1_r)
    token512_rouge2_r = np.array(token512_rouge2_r)
    token512_rougel_r = np.array(token512_rougel_r)
    chapter_text_length = np.array(chapter_text_length)
    print(f"avg chapter_text_length {np.mean(chapter_text_length)}")


    fig, axs = plt.subplots(3)
    fig.tight_layout(pad=2.5)
    fig.suptitle(f"valid/all video {len(rouge1_r)}/{len(all_vids)}")
    fig.subplots_adjust(top=0.85)
    axs[0].hist(rouge1_r, bins=20)
    axs[0].set_title("rouge1_recall", fontsize=10)
    axs[1].hist(rouge2_r, bins=20)
    axs[1].set_title("rouge2_recall", fontsize=10)
    axs[2].hist(rougel_r, bins=20)
    axs[2].set_title("rougeL_recall", fontsize=10)

    plt.savefig("./rouge_recall_distribution.jpg")
    plt.clf()


    fig, axs = plt.subplots(3)
    fig.tight_layout(pad=2.5)
    fig.suptitle(f"token512 in each chapter")
    fig.subplots_adjust(top=0.85)
    axs[0].hist(token512_rouge1_r, bins=20)
    axs[0].set_title("rouge1_recall", fontsize=10)
    axs[1].hist(token512_rouge2_r, bins=20)
    axs[1].set_title("rouge2_recall", fontsize=10)
    axs[2].hist(token512_rougel_r, bins=20)
    axs[2].set_title("rougeL_recall", fontsize=10)

    plt.savefig("./token512_rouge_recall_distribution.jpg")
    plt.clf()


    plt.hist(chapter_text_length, bins=20)
    plt.savefig("./avg_chapter_text_length.jpg")

