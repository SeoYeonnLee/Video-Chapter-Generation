import glob
from dataset_process_scripts.load_dataset_utils import load_dataset_with_subtitle, extract_first_timestamp


if __name__ == "__main__":
    # csv_file = "../dataset/top steam games/data.csv"
    # vids, titles, timestamp = parse_csv_to_list(csv_file)
    #
    # print(vids)
    # print(titles)
    # print(timestamp)

    asr_files = glob.glob("../dataset/how to cook steak/*0WG4YWZ0vBU.json")
    vids_with_asr, titles_with_asr, timestamps_with_asr, subtitles = load_dataset_with_subtitle(asr_files)
    for i in range(len(vids_with_asr)):
        vid = vids_with_asr[i]
        title = titles_with_asr[i]
        timestamp = timestamps_with_asr[i]
        subtitle = subtitles[i]

        # extract timestamp
        timepoint_secs = []
        descriptions = []
        for line in timestamp:
            sec, description = extract_first_timestamp(line)
            timepoint_secs.append(sec)
            descriptions.append(description)

        # find subtitle near by timestamp
        timepoint_idx = 0
        time_gap = 4
        text_near_timepoint = dict()
        for sub in subtitle:
            text = sub["text"]
            start = sub["start"]

            if start > timepoint_secs[timepoint_idx] + time_gap:
                timepoint_idx += 1

            if timepoint_idx >= len(timepoint_secs):
                break

            if timepoint_secs[timepoint_idx] - time_gap < start < timepoint_secs[timepoint_idx] + time_gap:
                if timepoint_idx in text_near_timepoint:
                    text_near_timepoint[timepoint_idx] += " " + text
                else:
                    text_near_timepoint[timepoint_idx] = text

        print(vid)
        for i in range(len(descriptions)):
            time_sec = timepoint_secs[i]
            description = descriptions[i]
            if i in text_near_timepoint:
                text = text_near_timepoint[i]
            else:
                text = ""

            view_link = f"https://www.youtube.com/watch?v={vid}&t={time_sec}s"
            print(view_link + "  " + description + " --> " + text)

        print()
