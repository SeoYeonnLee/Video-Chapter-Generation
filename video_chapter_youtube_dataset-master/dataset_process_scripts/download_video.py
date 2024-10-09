import multiprocessing
import multiple_process_utils
import os, glob
import youtube_dl
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list


def download_youtube_video(save_dir, all_vids):
    for vid in all_vids:
        if os.path.exists(os.path.join(save_dir, f"{vid}.mp4")):
            continue

        save_filename = os.path.join(save_dir, f"{vid}.mp4")
        print(f"download {vid}...")
        link = "https://www.youtube.com/watch?v=" + vid
        try:
            # YouTube library may download fail
            # yt = YouTube(link)
            # yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first().download(output_path=save_dir, filename=vid)

            ydl_opts = {
                'outtmpl': save_filename,
                "format": "18"
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                dictMeta = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
                duration = dictMeta["duration"]
                if duration < 1800:         # only consider video which is less than 30 minutes
                    ydl.download([link])

        except Exception as e:
            print(f"Exception to download {vid}")
            print(f"{e}")
            print()

def multiple_process_run(save_dir, all_vids, process_num=10):
    pool = multiprocessing.Pool(process_num)
    chunked_data = multiple_process_utils.split_data(process_num, all_vids)
    for i, d in enumerate(chunked_data):
        pool.apply_async(download_youtube_video,
                         args=(save_dir, d),
                         error_callback=multiple_process_utils.subprocess_print_err)

    print('Waiting for all subprocesses done...')
    pool.close()
    pool.join()
    print('All subprocesses done.')


if __name__ == "__main__":
    # debug
    # download_youtube_video(save_dir="C:/Users/Admin/Desktop/temp_audio", all_vids=["-IJuKT1mHO8"])
    # print()

    video_save_dir = "/opt/tiger/"
    all_csv_file = glob.glob("/opt/tiger/video_chapter_youtube_dataset/dataset/*/data.csv")
    all_vids = list()
    for csv_file in all_csv_file:
        vids, _, _ = parse_csv_to_list(csv_file, w_duration=False)
        all_vids.extend(vids)

    all_vids = list(set(all_vids))
    need_to_download_vids = []
    for vid in all_vids:
        if os.path.exists(os.path.join(video_save_dir, f"{vid}.mp4")):
            continue
        need_to_download_vids.append(vid)

    print(f"{len(all_vids) - len(need_to_download_vids)}/{len(all_vids)}, {len(need_to_download_vids)} videos need to be downloaded")

    download_youtube_video(video_save_dir, all_vids)
    # multiple_process_run(video_save_dir, need_to_download_vids)

