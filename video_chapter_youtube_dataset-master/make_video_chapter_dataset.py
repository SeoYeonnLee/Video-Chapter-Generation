"""
Collect video chapter dataset from youtube.
It uses 3 api
1. youtube searching
2. youtube video description for timestamps
3. transcript (asr subtitle)

1, 2 based on official api
3 based on https://github.com/jdepoix/youtube-transcript-api


"""


from collections import defaultdict
from youtube_transcript_api import YouTubeTranscriptApi
import multiple_process_utils
import re
import os, glob
import multiprocessing
import time
import json
import pandas as pd
import requests

# API_KEY = "AIzaSyB-wUr9YQt8LOwlq7B19oJr7ViFI5VA-L0"   # ranger.lcy@gmail.com
API_KEY = "AIzaSyDX3I3GvHWsupsOUurtvDwMjLgxL_1FQGo"     # lecanyu@gmail.com
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"

TIMESTAMP_DELIMITER = "%^&*"


def save_result(videos, search_response):
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            videos["title"].append(search_result["snippet"]["title"])
            videos["description"].append(search_result["snippet"]["description"])
            videos["videoId"].append(search_result["id"]["videoId"])
            videos["publishedAt"].append(search_result["snippet"]["publishedAt"])
            videos["channelId"].append(search_result["snippet"]["channelId"])
            # videos["channelTitle"].append(search_result["snippet"]["channelTitle"])
            # videos["liveBroadcastContent"].append(search_result["snippet"]["liveBroadcastContent"])


def parse_timestamp(description):
    timestamp_lines = []

    lines = description.split("\n")
    for i, line in enumerate(lines):
        if len(line) > 150:
            continue

        if len(timestamp_lines) == 0 and "0:00" in line:
            line = re.sub(r'http\S+', '', line)  # remove all http urls
            timestamp_lines.append(line)
            continue

        if len(timestamp_lines) > 0:
            if re.search("\d{1}:\d{2}", line):
                line = re.sub(r'http\S+', '', line)     # remove all http urls
                timestamp_lines.append(line)

    return timestamp_lines


def subprocess_request_video_description(process_idx, chunked_vids, contain_timestamp_index_dict, timestamps_dict):
    video_params = {
        "part": "snippet",
        "key": API_KEY
    }
    contain_timestamp_index = []
    timestamps = []
    for ele in chunked_vids:
        original_index, vid = ele
        video_params["id"] = vid
        r = requests.get(YOUTUBE_VIDEO_URL + "?", params=video_params)
        if r.status_code == 200:
            data = r.json()
            description = data["items"][0]["snippet"]["description"]
            timestamp_lines = parse_timestamp(description)

            if len(timestamp_lines) > 0:
                contain_timestamp_index.append(original_index)
                timestamps.append(timestamp_lines)

    contain_timestamp_index_dict[process_idx] = contain_timestamp_index
    timestamps_dict[process_idx] = timestamps


def subprocess_request_asr(process_idx, chunked_vids, asr_subtitle_dict):
    asr_subtitles = []
    for ele in chunked_vids:
        original_index, vid = ele
        # get asr caption
        try:
            # manual_transcript = transcript_list.find_manually_created_transcript(['en'])
            # manual_result = manual_transcript.fetch()
            transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
            auto_transcript = transcript_list.find_generated_transcript(['en'])

            auto_result = auto_transcript.fetch()
            asr_subtitles.append([original_index, auto_result])
        except Exception as e:
            # May not be able to get transcript
            # Could not retrieve a transcript for the video https://www.youtube.com/watch?v=W_i5fu-ajhA!
            # This is most likely caused by:
            # Subtitles are disabled for this video
            asr_subtitles.append([original_index, []])

    asr_subtitle_dict[process_idx] = asr_subtitles


def search_youtube_video(searchTerm, maxResults, run_type="sequential"):
    videos = defaultdict(list)
    publish_after = "2020-05-01T00:00:00Z"
    search_params = {
        'q': searchTerm + " timestamp",     # append 'timestamp' keyword to get more relevant results
        'part': 'id,snippet',
        'maxResults': maxResults,
        'key': API_KEY,
        'publishedAfter': publish_after
    }

    """
    Send search request
    """
    print(f"Send search request...")
    r = requests.get(YOUTUBE_SEARCH_URL + "?", params=search_params)
    if r.status_code != 200:
        if "quota" in r.text:
            print("Exceed quota. Exit...")
            exit()
        return None

    search_response = r.json()
    nextPageToken = search_response.get("nextPageToken")
    save_result(videos, search_response)
    print(f"current search result size {len(videos['videoId'])}")

    while len(videos["videoId"]) < maxResults:
        search_params.update({'pageToken': nextPageToken})
        r = requests.get(YOUTUBE_SEARCH_URL + "?", params=search_params)
        if r.status_code != 200:
            return None

        search_response = r.json()
        nextPageToken = search_response.get("nextPageToken")
        save_result(videos, search_response)
        print(f"current search result size {len(videos['videoId'])}")

    """
    Send video request to get complete description, so that we can get timestamps
    """
    print(f"Send video request to get complete description...")
    if run_type == "sequential":
        # sequentially run
        contain_timestamp_index = []
        timestamps = []
        video_params = {
            "part": "snippet",
            "key": API_KEY
        }
        for i, video_id in enumerate(videos["videoId"]):
            video_params["id"] = video_id
            r = requests.get(YOUTUBE_VIDEO_URL + "?", params=video_params)
            if r.status_code == 200:
                data = r.json()
                description = data["items"][0]["snippet"]["description"]
                timestamp_lines = parse_timestamp(description)
                if len(timestamp_lines) > 0:
                    contain_timestamp_index.append(i)
                    timestamps.append(timestamp_lines)
    else:
        # multiple process run
        process_num = 8
        pool = multiprocessing.Pool(process_num)
        contain_timestamp_index_dict = multiprocessing.Manager().dict()
        timestamps_dict = multiprocessing.Manager().dict()
        original_index_vid_pairs = [[i, vid] for i, vid in enumerate(videos["videoId"])]

        chunked_data = multiple_process_utils.split_data(process_num, original_index_vid_pairs)
        for i, d in enumerate(chunked_data):
            pool.apply_async(subprocess_request_video_description, args=(i, d, contain_timestamp_index_dict, timestamps_dict), error_callback=multiple_process_utils.subprocess_print_err)

        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()
        print('All subprocesses done.')

        contain_timestamp_index = []
        timestamps = []
        for k, v in contain_timestamp_index_dict.items():
            contain_timestamp_index.extend(v)
            timestamps.extend(timestamps_dict[k])

    """
    Send request to get asr subtitle
    """
    print(f"Send request to get asr subtitle...")
    if run_type == "sequential":
        # sequentially run
        asr_subtitles = []
        vid_has_timestamp = [videos["videoId"][index] for index in contain_timestamp_index]
        for vid in vid_has_timestamp:
            # get asr caption
            try:
                # manual_transcript = transcript_list.find_manually_created_transcript(['en'])
                # manual_result = manual_transcript.fetch()
                transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
                auto_transcript = transcript_list.find_generated_transcript(['en'])

                auto_result = auto_transcript.fetch()
                asr_subtitles.append(auto_result)
            except Exception as e:
                # May not be able to get transcript
                # Could not retrieve a transcript for the video https://www.youtube.com/watch?v=W_i5fu-ajhA!
                # This is most likely caused by:
                # Subtitles are disabled for this video
                asr_subtitles.append([])
    else:
        # multiple process run
        process_num = 8
        pool = multiprocessing.Pool(process_num)
        asr_subtitle_dict = multiprocessing.Manager().dict()
        vid_has_timestamp = [[i, videos["videoId"][index]] for i, index in enumerate(contain_timestamp_index)]
        if len(vid_has_timestamp) > 0:
            chunked_data = multiple_process_utils.split_data(process_num, vid_has_timestamp)
            for i, d in enumerate(chunked_data):
                pool.apply_async(subprocess_request_asr,
                                 args=(i, d, asr_subtitle_dict),
                                 error_callback=multiple_process_utils.subprocess_print_err)

            print('Waiting for all subprocesses done...')
            pool.close()
            pool.join()
            print('All subprocesses done.')
        asr_subtitle_list = []
        for k, v in asr_subtitle_dict.items():
            asr_subtitle_list.extend(v)
        asr_subtitle_list = sorted(asr_subtitle_list, key=lambda x: x[0])
        asr_subtitles = [x[1] for x in asr_subtitle_list]

    """
    Organize data
    """
    print(f"Organize data...")
    videos_has_timestamp = defaultdict(list)
    for index, videos_i in enumerate(contain_timestamp_index):
        videos_has_timestamp["videoId"].append(videos["videoId"][videos_i])
        videos_has_timestamp["title"].append(videos["title"][videos_i])
        videos_has_timestamp["subtitle"].append(asr_subtitles[index])

        # convert timestamp list to string
        timestamp = timestamps[index]
        timestamp = TIMESTAMP_DELIMITER.join(timestamp)
        videos_has_timestamp["timestamp"].append(timestamp)

    return videos_has_timestamp


def save_to_file(query, videos_has_timestamp):
    save_dir = f"./dataset/{query}"
    os.makedirs(save_dir, exist_ok=True)

    vid_list = videos_has_timestamp["videoId"]
    subtitle_list = videos_has_timestamp.pop("subtitle")

    for vid, sub in zip(vid_list, subtitle_list):
        if len(sub) > 10:
            subtitle_file = os.path.join(save_dir, f"subtitle_{vid}.json")
            with open(subtitle_file, "w") as f:
                json.dump(sub, f)

    save_data_file_path = os.path.join(save_dir, "data.csv")
    data = pd.DataFrame(videos_has_timestamp)
    data.to_csv(save_data_file_path)


def rule_out_exist_category(list_before_rule_out):
    file_or_dirs = glob.glob("./dataset/*")
    file_or_dirs = [os.path.basename(x) for x in file_or_dirs]

    s1 = set(list_before_rule_out)
    s2 = set(file_or_dirs)
    list_after_rule_out = s1.difference(s2)
    list_after_rule_out = list(list_after_rule_out)
    return list_after_rule_out


if __name__ == "__main__":
    # run a query for debug
    # query = "top steam games"
    # result_num = 20
    # st = time.time()
    # videos_has_timestamp = search_youtube_video(query, result_num, run_type="sequential")
    # et = time.time()
    # print(f"cost time {et - st}")
    #
    # print(f"total size {len(videos_has_timestamp['videoId'])}")
    # with_subtitle_num = 0
    # for sub in videos_has_timestamp["subtitle"]:
    #     if len(sub) > 0:
    #         with_subtitle_num += 1
    # print(f"total size with subtitle {with_subtitle_num}")
    #
    # for title, vid, desc, subt in zip(videos_has_timestamp["title"], videos_has_timestamp["videoId"], videos_has_timestamp["timestamp"], videos_has_timestamp["subtitle"]):
    #     print(vid)
    #     print(title)
    #     print(desc)
    #     print(subt)
    #     print("===========================================")

    # save_to_file(query, videos_has_timestamp)


    # run a batch of queries
    # with open("./manual_search_query.txt", "r") as f:
    with open("./wikihow_query.txt", "r") as f:
        lines = [x.strip() for x in f.readlines()]
        lines = [x for x in lines if x != ""]
    
    # rule out existed results
    lines = rule_out_exist_category(lines)
    
    result_num = 200
    total_size = 0
    with_subtitle_size = 0
    for i, query in enumerate(lines):
        st = time.time()
        videos_has_timestamp = search_youtube_video(query, result_num, run_type="parallel")
        et = time.time()
        if videos_has_timestamp is None:
            continue
        print(f"cost time {et - st}")
    
        print(f"total size {len(videos_has_timestamp['videoId'])}")
        total_size += len(videos_has_timestamp['videoId'])
    
        with_subtitle_num = 0
        for sub in videos_has_timestamp["subtitle"]:
            if len(sub) > 0:
                with_subtitle_num += 1
        print(f"total size with subtitle {with_subtitle_num}")
        with_subtitle_size += with_subtitle_num
    
        print(f"All queries {result_num * (i+1)}")
        print(f"All total size {total_size}")
        print(f"All total size with subtitle {with_subtitle_size}")
    
        save_to_file(query, videos_has_timestamp)

