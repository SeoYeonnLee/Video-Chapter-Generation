import multiprocessing
import multiple_process_utils
import os
import glob
import yt_dlp  # 변경된 부분
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list


def download_youtube_video(save_dir, all_vids):
    for vid in all_vids:
        video_path = os.path.join(save_dir, f"{vid}.mp4")
        if os.path.exists(video_path):
            # 이미 다운로드된 경우 건너뜁니다.
            continue

        print(f"Downloading {vid}...")
        link = "https://www.youtube.com/watch?v=" + vid
        try:
            ydl_opts = {
                'outtmpl': video_path,
                "format": "18"  # 360p 화질로 다운로드 (변경 가능)
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # 변경된 부분
                # 메타 정보만 먼저 가져오기 (다운로드 안 함)
                dictMeta = ydl.extract_info(link, download=False)
                duration = dictMeta.get("duration", 0)

                # 30분 미만인 비디오만 다운로드
                if duration < 1800:
                    ydl.download([link])

        except Exception as e:
            print(f"Exception occurred while downloading {vid}: {e}\n")


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
    video_save_dir = "./vids"
    os.makedirs(video_save_dir, exist_ok=True)  # 비디오 저장 디렉토리 생성

    # 모든 CSV 파일에서 비디오 ID 목록 가져오기
    all_csv_files = glob.glob("./dataset/*/data.csv")
    all_vids = list()
    for csv_file in all_csv_files:
        vids, _, _ = parse_csv_to_list(csv_file, w_duration=False)
        all_vids.extend(vids)

    # 중복 제거
    all_vids = list(set(all_vids))

    # 이미 다운로드된 비디오를 제외한 목록 생성
    need_to_download_vids = [vid for vid in all_vids if not os.path.exists(os.path.join(video_save_dir, f"{vid}.mp4"))]

    print(f"{len(all_vids) - len(need_to_download_vids)}/{len(all_vids)} videos are already downloaded.")
    print(f"{len(need_to_download_vids)} videos need to be downloaded.")

    # 멀티프로세싱을 사용하여 비디오 다운로드
    if need_to_download_vids:
        multiple_process_run(video_save_dir, need_to_download_vids, process_num=10)


'''import multiprocessing
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

'''