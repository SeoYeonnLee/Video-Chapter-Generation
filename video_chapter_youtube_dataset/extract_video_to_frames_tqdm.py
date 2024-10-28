import os, glob
import shutil
import pandas as pd
import multiprocessing
import multiple_process_utils
from tqdm import tqdm

def extract_frame_fn(process_idx, video_files, durations, video_frame_dir, extract_fps=1, progress_queue=None):
    for i, video_file in enumerate(video_files):
        if durations is not None and durations[i] > 1800:
            if progress_queue:
                progress_queue.put(1)
            continue
        vid = os.path.basename(video_file).split(".")[0]
        frame_save_dir = os.path.join(video_frame_dir, vid)
        if os.path.exists(frame_save_dir):
            if durations is not None:
                img_num = len(glob.glob(frame_save_dir + "/*.jpg"))
                if img_num < int(durations[i]) - 1:
                    shutil.rmtree(frame_save_dir)
                else:
                    if progress_queue:
                        progress_queue.put(1)
                    continue
            else:
                shutil.rmtree(frame_save_dir)
        os.makedirs(frame_save_dir)

        save_path = frame_save_dir + "/%05d.jpg"
        os.system(f"ffmpeg -i {video_file} -s 224x224 -r {extract_fps} {save_path}")
        
        if progress_queue:
            progress_queue.put(1)

def progress_tracker(queue, total):
    pbar = tqdm(total=total, desc="Extracting frames")
    while True:
        item = queue.get()
        if item is None:
            break
        pbar.update(item)
    pbar.close()

if __name__ == "__main__":
    data = pd.read_csv("./dataset/all_in_one_with_subtitle_new.csv")
    vids = list(data["videoId"].values)
    durations = list(data["duration"].values)
    video_dir = "vids"
    video_files = [os.path.join(video_dir, f"{x}.mp4") for x in vids]

    video_frame_dir = "./youtube_video_frame_dataset"
    os.makedirs(video_frame_dir, exist_ok=True)

    extract_fps = 1

    # multiple process
    process_num = 8
    pool = multiprocessing.Pool(process_num)
    
    # 진행 상황 추적을 위한 큐 생성
    progress_queue = multiprocessing.Manager().Queue()

    # 진행 상황 추적 프로세스 시작
    tracker = multiprocessing.Process(target=progress_tracker, args=(progress_queue, len(video_files)))
    tracker.start()

    chunked_data = multiple_process_utils.split_data(process_num, video_files)
    chunked_duration_data = multiple_process_utils.split_data(process_num, durations)
    for i, d in enumerate(chunked_data):
        chunked_dura = chunked_duration_data[i]
        pool.apply_async(extract_frame_fn,
                         args=(i, d, chunked_dura, video_frame_dir, extract_fps, progress_queue),
                         error_callback=multiple_process_utils.subprocess_print_err)

    print('Waiting for all subprocesses done...')
    pool.close()
    pool.join()

    # 진행 상황 추적 종료
    progress_queue.put(None)
    tracker.join()

    print('All subprocesses done.')