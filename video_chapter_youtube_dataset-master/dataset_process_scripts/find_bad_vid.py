"""
find wrong data

"""

import pandas as pd
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list, extract_first_timestamp


def find_timestamp_too_close(all_timestamps, time_gap=8):
    bad_indices = []
    for i, timestamps in enumerate(all_timestamps):
        timepoint_secs = []
        descriptions = []
        for line in timestamps:
            sec, description = extract_first_timestamp(line)

            if len(timepoint_secs) > 0:
                if sec - timepoint_secs[-1] < time_gap:
                    bad_indices.append(i)
                    break

            timepoint_secs.append(sec)
            descriptions.append(description)

    return bad_indices 


def find_duration_too_short(durations, threshold=100):
    bad_indices = []
    for i, duration in enumerate(durations):
        if duration < threshold:
            bad_indices.append(i)

    return bad_indices 


if __name__ == "__main__":
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    all_vids, all_titles, all_durations, all_timestamps = parse_csv_to_list(data_file)

    # a = find_timestamp_too_close(all_timestamps, 10)

    bad_indices = find_timestamp_too_close(all_timestamps) + find_duration_too_short(all_durations)
    bad_vids = [all_vids[i] for i in bad_indices]
    print(len(bad_vids))
    print(bad_vids)


