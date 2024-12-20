"""
flat testing videos to clips, so that we can quickly test model performance without reloading dataloader many times

"""

import json
import os, glob
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list, extract_first_timestamp
from tqdm import tqdm


def flat_videos2clips(img_dir, data_file, test_vid_file, clip_frame_num=12):
    half_clip_frame_num = int(clip_frame_num // 2)

    # processed vids
    with open(test_vid_file, "r") as f:
        vids = f.readlines()
        vids = [x.strip() for x in vids]

    # basic data
    all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)

    # subtitles
    asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
    asr_files = dict()
    for asr_file in asr_file_list:
        filename = os.path.basename(asr_file)
        vid = filename.split(".")[0][9:]
        asr_files[vid] = asr_file

    all_clip_infos = []
    
    for idx, vid in enumerate(tqdm(vids, desc="Processing videos")):

        i = all_vids.index(vid)
        title = titles[i]
        duration = durations[i]
        timestamp = timestamps[i]

        asr_file = asr_files[vid]
        with open(asr_file, "r") as f:
            subtitles = json.load(f)
        
        # image num
        image_path = os.path.join(img_dir, vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))

        # gt cut points
        cut_points = []    # record G.T. cut point
        descriptions = []
        for timestamp_str in timestamp:
            gap = 0
            sec, description = extract_first_timestamp(timestamp_str)
            if sec < 4:
                continue
            if sec > image_num - 4:
                continue
            
            cut_points.append(sec)
            descriptions.append(description)

        # go through all clips within this video
        max_offset = 2
        clips = [[start_t, start_t + clip_frame_num] for start_t in range(0, image_num - clip_frame_num, 2 * max_offset)]
        batch_num = len(clips)
        for batch_i in range(batch_num):
            # this clip's start and end time
            clip_start_sec, clip_end_sec = clips[batch_i]

            # label is determined by IoU
            label = 0
            for cp in cut_points:
                pos_st = cp - half_clip_frame_num
                pos_et = cp + half_clip_frame_num
                a = max(clip_start_sec, pos_st)
                mi = min(clip_start_sec, pos_st)
                b = min(clip_end_sec, pos_et)
                ma = max(clip_end_sec, pos_et)

                iou = (b - a) / (ma - mi) 
                if iou >= (clip_frame_num - max_offset) / (clip_frame_num + max_offset):
                    label = 1

            # get the subtitle in-between [clip_start_sec - text_extra_time_gap, clip_end_sec + text_extra_time_gap]
            text_extra_time_gap = 1
            text_clip = ""
            for sub in subtitles:
                text = sub["text"]
                start_sec = sub["start"]
                if clip_start_sec - text_extra_time_gap < start_sec < clip_end_sec + text_extra_time_gap:
                    if len(text_clip) == 0:
                        text_clip += text
                    else:
                        text_clip += " " + text
            
            # get image paths in this clip
            img_path_list = []
            for idx in range(clip_start_sec, clip_end_sec):
                # There is a bug ffmpeg extract frame, which causes the image misalign frame. 
                # We offset right frame 2 unit, but keep the beginning and the end not change.  
                if clip_start_sec <= 2 or clip_start_sec >= image_num - clip_frame_num - 2:
                    image_filename = "%05d.jpg"%(idx+1)
                else:
                    image_filename = "%05d.jpg"%(idx+3)
                image_filename = os.path.join(image_path, image_filename)
                img_path_list.append(image_filename)


            clip_info = {
                "image_paths": img_path_list,
                "text_clip": text_clip,
                "clip_label": label,
                "clip_start_end": [clip_start_sec, clip_end_sec],
                "cut_points": cut_points,
                "vid": vid
            }
            all_clip_infos.append(clip_info)
    
    return all_clip_infos
    


# with open("/opt/tiger/video_chapter_youtube_dataset/dataset/test_easy_clips.json", "r") as f:
#     data = json.load(f)
#     vids = [x["vid"] for x in data]
#     print(set(vids))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--clip_frame_num', default=12, type=int)
    args = parser.parse_args()

    img_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/youtube_video_frame_dataset"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_final.csv"
    # test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"
    # test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_test.txt"
    # train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_train.txt"
    test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_validation.txt"


    clip_frame_num = args.clip_frame_num
    all_clip_infos = flat_videos2clips(img_dir, data_file, test_vid_file, clip_frame_num)

    # save all test clips
    # save_json_file = f"/opt/tiger/video_chapter_youtube_dataset/dataset/test_clips_clip_frame_num_{clip_frame_num}.json"
    save_json_file = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/validation_clips_clip_frame_num_{clip_frame_num}.json"
    # save_json_file = f"/opt/tiger/video_chapter_youtube_dataset/dataset/all_clips_clip_frame_num_{clip_frame_num}.json"
    with open(save_json_file, "w") as f:
        json.dump(all_clip_infos, f)

    # save as easy, hard ...
    # easy_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/easy_vid.txt"
    # hard_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/hard_vid.txt"
    # ambiguous_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/ambiguous_vid.txt"
    # wrong_data_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/wrong_data_vid.txt"
    # with open(easy_vid_file, "r") as f:
    #     vids = f.readlines()
    #     easy_vids = [x.strip() for x in vids]
    # with open(hard_vid_file, "r") as f:
    #     vids = f.readlines()
    #     hard_vids = [x.strip() for x in vids]
    # with open(ambiguous_vid_file, "r") as f:
    #     vids = f.readlines()
    #     ambiguous_vids = [x.strip() for x in vids]
    # with open(wrong_data_vid_file, "r") as f:
    #     vids = f.readlines()
    #     wrong_data_vids = [x.strip() for x in vids]

    # save_easy_json_file = f"/opt/tiger/video_chapter_youtube_dataset/dataset/test_easy_clips_clip_frame_num_{clip_frame_num}.json"
    # save_hard_json_file = f"/opt/tiger/video_chapter_youtube_dataset/dataset/test_hard_clips_clip_frame_num_{clip_frame_num}.json"
    # save_ambiguous_json_file = f"/opt/tiger/video_chapter_youtube_dataset/dataset/test_ambiguous_clips_clip_frame_num_{clip_frame_num}.json"
    # save_wrong_data_json_file = f"/opt/tiger/video_chapter_youtube_dataset/dataset/test_wrong_data_clips_clip_frame_num_{clip_frame_num}.json"

    # easy_clip_infos = []
    # hard_clip_infos = []
    # ambiguous_clip_infos = []
    # wrong_data_clip_infos = []
    # for clip_info in all_clip_infos:
    #     if clip_info["vid"] in easy_vids:
    #         easy_clip_infos.append(clip_info)
    #     if clip_info["vid"] in hard_vids:
    #         hard_clip_infos.append(clip_info)
    #     if clip_info["vid"] in ambiguous_vids:
    #         ambiguous_clip_infos.append(clip_info)
    #     if clip_info["vid"] in wrong_data_vids:
    #         wrong_data_clip_infos.append(clip_info)

    # with open(save_easy_json_file, "w") as f:
    #     json.dump(easy_clip_infos, f)
    # with open(save_hard_json_file, "w") as f:
    #     json.dump(hard_clip_infos, f)
    # with open(save_ambiguous_json_file, "w") as f:
    #     json.dump(ambiguous_clip_infos, f)
    # with open(save_wrong_data_json_file, "w") as f:
    #     json.dump(wrong_data_clip_infos, f)