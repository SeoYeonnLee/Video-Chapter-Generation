"""
Combine all youtube categories to one dataset

"""

import glob, os
import cv2
import pandas as pd
import multiple_process_utils
import multiprocessing
from dataset_process_scripts import load_dataset_utils
from make_video_chapter_dataset import TIMESTAMP_DELIMITER


def multiple_process_load_video(process_idx, paths, vid_dict, duration_dict):
    vids = []
    durations = []
    for i, path in enumerate(paths):
        print(f"process {process_idx}, load video {i}/{len(paths)}...")
        vid = os.path.basename(path).split(".")[0]
        vidcap = cv2.VideoCapture(path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = round(frame_count / fps, 2)
            vids.append(vid)
            durations.append(duration)
        else:
            # if video file is invalid, delete it
            os.remove(path)

    vid_dict[process_idx] = vids
    duration_dict[process_idx] = durations



def combine_all_data_with_subtitle():
    video_root = "D:/youtube_video_dataset"
    save_data_file_path = "../dataset/all_in_one_with_subtitle.csv"
    asr_files = glob.glob("../dataset/*/subtitle_*.json")

    # multiple process run
    all_video_files = glob.glob(video_root + "/*.mp4")

    process_num = 8
    pool = multiprocessing.Pool(process_num)
    vid_dict = multiprocessing.Manager().dict()
    duration_dict = multiprocessing.Manager().dict()
    chunked_data = multiple_process_utils.split_data(process_num, all_video_files)
    for i, d in enumerate(chunked_data):
        pool.apply_async(multiple_process_load_video,
                         args=(i, d, vid_dict, duration_dict),
                         error_callback=multiple_process_utils.subprocess_print_err)

    print('Waiting for all subprocesses done...')
    pool.close()
    pool.join()
    print('All subprocesses done.')

    vid2duration = dict()
    saved_vids = []
    durations = []
    for k, v in vid_dict.items():
        saved_vids.extend(v)
    for k, v in duration_dict.items():
        durations.extend(v)
    assert len(saved_vids) == len(durations), "len(saved_vids) == len(durations) is not satisfied"
    for vid, duration in zip(saved_vids, durations):
        vid2duration[vid] = duration

    # load all data with subtitle
    vids_with_asr, titles_with_asr, timestamps_with_asr, subtitles = load_dataset_utils.load_dataset_with_subtitle(asr_files)
    dataset = dict()
    for i in range(len(vids_with_asr)):
        vid = vids_with_asr[i]
        title = titles_with_asr[i]
        timestamp = timestamps_with_asr[i]
        subtitle = subtitles[i]

        if vid not in vid2duration:
            continue
        if vid2duration[vid] > 1800:    # only consider video which is less than 30 minutes
            continue

        all_text = ""
        for x in subtitle:
            all_text += x["text"]
        all_text = all_text.split(" ")
        duration = vid2duration[vid]
        if len(all_text) / duration < 0.5:  # speak 0.5 word/second
            continue
        if len(timestamp) < 3:
            continue

        sec, description = load_dataset_utils.extract_first_timestamp(timestamp[0])
        if sec > 0:
            continue
        dataset[vid] = [vid, title, duration, timestamp]

    vids = []
    titles = []
    durations = []
    timestamps = []
    for k, v in dataset.items():
        vid, title, duration, timestamp = v
        vids.append(vid)
        titles.append(title)
        durations.append(duration)

        # convert timestamp list to string
        timestamp = TIMESTAMP_DELIMITER.join(timestamp)
        timestamps.append(timestamp)

    all_in_one = {
        "videoId": vids,
        "title": titles,
        "duration": durations,
        "timestamp": timestamps
    }
    data = pd.DataFrame(all_in_one)
    data.to_csv(save_data_file_path)



if __name__ == "__main__":
    # combine_all_data()
    combine_all_data_with_subtitle()
