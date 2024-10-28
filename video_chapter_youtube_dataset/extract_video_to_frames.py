import os, glob
import shutil
import pandas as pd
import multiprocessing
import multiple_process_utils


""" extract all video """
def extract_frame_fn(process_idx, video_files, durations, video_frame_dir, extract_fps=1):
    for i, video_file in enumerate(video_files):
        if durations is not None and durations[i] > 1800:
            continue
        vid = os.path.basename(video_file).split(".")[0]
        print(f"process {process_idx}, extract video {i}/{len(video_files)}, {vid}...")
        frame_save_dir = os.path.join(video_frame_dir, vid)
        if os.path.exists(frame_save_dir):
            if durations is not None:
                img_num = len(glob.glob(frame_save_dir + "/*.jpg"))
                if img_num < int(durations[i]) - 1:
                    shutil.rmtree(frame_save_dir)
                else:
                    continue
            else:
                shutil.rmtree(frame_save_dir)
        os.makedirs(frame_save_dir)

        save_path = frame_save_dir + "/%05d.jpg"
        os.system(f"ffmpeg -i {video_file} -s 224x224 -r {extract_fps} {save_path}")


if __name__ == "__main__":
    data = pd.read_csv("./dataset/all_in_one_with_subtitle_new.csv")
    vids = list(data["videoId"].values)
    durations = list(data["duration"].values)
    video_dir = "vids"
    video_files = [video_dir + "/" + x + ".mp4" for x in vids]

    video_frame_dir = "./youtube_video_frame_dataset"
    os.makedirs(video_frame_dir, exist_ok=True)

    extract_fps = 1

    # single process
    # extract_frame_fn(0, video_files, video_frame_dir, extract_fps)

    # multiple process
    process_num = 8
    pool = multiprocessing.Pool(process_num)
    chunked_data = multiple_process_utils.split_data(process_num, video_files)
    chunked_duration_data = multiple_process_utils.split_data(process_num, durations)
    for i, d in enumerate(chunked_data):
        chunked_dura = chunked_duration_data[i]
        pool.apply_async(extract_frame_fn,
                         args=(i, d, chunked_dura, video_frame_dir, extract_fps),
                         error_callback=multiple_process_utils.subprocess_print_err)

    print('Waiting for all subprocesses done...')
    pool.close()
    pool.join()
    print('All subprocesses done.')




    """ extract for mini dataset """
    # data = pd.read_csv("D:/py3_code/video_chapter_youtube_dataset/dataset/test_mini_dataset.csv")
    # vids = list(data["videoId"].values)
    # video_dir = "D:/youtube_video_dataset"
    # video_files = [video_dir + "/" + x + ".mp4" for x in vids]
    #
    # video_frame_dir = "D:/youtube_video_frame_minidataset"
    # os.makedirs(video_frame_dir, exist_ok=True)
    #
    # extract_fps = 1
    # extract_frame_fn(0, video_files, None, video_frame_dir, extract_fps)
