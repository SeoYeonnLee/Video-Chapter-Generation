import pandas as pd
import os
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list, extract_first_timestamp
from make_video_chapter_dataset import TIMESTAMP_DELIMITER

"""
find wrong data & delete bad vid (wrong data) from a list

"""


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
    data_file = "./dataset/all_in_one_with_subtitle.csv"
    new_data_file = "./dataset/all_in_one_with_subtitle_new.csv"

    all_vids, all_titles, all_durations, all_timestamps = parse_csv_to_list(data_file)

    # find_bad_vid
    bad_indices = find_timestamp_too_close(all_timestamps) + find_duration_too_short(all_durations)
    bad_vids = [all_vids[i] for i in bad_indices]
    print(f'bad videos len: {len(bad_vids)}')
    print(bad_vids)


    # clean_bad_vid
    vids = []
    titles = []
    durations = []
    timestamps = []
    for i in range(len(all_vids)):
        vid = all_vids[i]

        # bad vid이거나 이미 해당 비디오 데이터 존재하면 continue
        if vid in bad_vids or vid in vids:
            continue

        vids.append(vid)
        titles.append(all_titles[i])
        durations.append(all_durations[i])

        # convert timestamp list to string
        timestamp = all_timestamps[i]
        timestamp = TIMESTAMP_DELIMITER.join(timestamp)
        timestamps.append(timestamp)

    all_in_one = {
        "videoId": vids,
        "title": titles,
        "duration": durations,
        "timestamp": timestamps
    }
    data = pd.DataFrame(all_in_one)
    data.to_csv(new_data_file)
    print('clean bad videos completed.')