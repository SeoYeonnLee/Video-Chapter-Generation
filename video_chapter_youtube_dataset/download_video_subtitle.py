import multiprocessing
import multiple_process_utils
import os
import glob
import yt_dlp


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
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
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
    video_save_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/vids"
    # os.makedirs(video_save_dir, exist_ok=True)  # 비디오 저장 디렉토리 생성

    # 수정된 부분!!!
    # 모든 subtitle JSON 파일에서 비디오 ID 목록 가져오기
    all_json_files = glob.glob("./dataset/*/subtitle_*.json")
    all_vids = list()
    for json_file in all_json_files:
        vid = os.path.basename(json_file).replace('subtitle_', '').split(".")[0]  # 'subtitle_' 제거하고 확장자 제거
        all_vids.append(vid)

    # 중복 제거
    all_vids = list(set(all_vids))

    # 이미 다운로드된 비디오를 제외한 목록 생성
    need_to_download_vids = [vid for vid in all_vids if not os.path.exists(os.path.join(video_save_dir, f"{vid}.mp4"))]

    print(f"{len(all_vids) - len(need_to_download_vids)}/{len(all_vids)} videos are already downloaded.")
    print(f"{len(need_to_download_vids)} videos need to be downloaded.")

    # 멀티프로세싱을 사용하여 비디오 다운로드
    if need_to_download_vids:
        multiple_process_run(video_save_dir, need_to_download_vids, process_num=10)
