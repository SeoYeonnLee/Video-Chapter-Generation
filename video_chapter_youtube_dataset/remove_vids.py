import os
import pandas as pd
from tqdm import tqdm


# csv에 존재하는 비디오 아이디
data = pd.read_csv("/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/dataset/all_in_one_with_subtitle_final.csv")
vids = list(data["videoId"].values)


# vids 폴더 내의 전체 mp4 파일 리스트
video_folder = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/vids"
all_videos = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

# vids 리스트에 존재하는 비디오 파일
existing_videos = [vid for vid in all_videos if any(vid.startswith(v) for v in vids)]

# vids 리스트에 존재하지 않는 비디오 파일
non_existing_videos = list(set(all_videos) - set(existing_videos))

# 개수 출력
print(f"전체 비디오 파일 개수: {len(all_videos)}")
print(f"vids 리스트에 존재하는 비디오 파일 개수: {len(existing_videos)}")
print(f"vids 리스트에 존재하지 않는 비디오 파일 개수: {len(non_existing_videos)}")

# 삭제할 파일 리스트 txt 파일로 저장
txt_file_path = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/deleted_videos.txt"
with open(txt_file_path, "w") as f:
    for file in non_existing_videos:
        f.write(file + "\n")

print(f"삭제할 비디오 리스트 저장 완료: {txt_file_path}")

# vids 리스트에 존재하지 않는 비디오 파일 삭제
for file in tqdm(non_existing_videos, desc="삭제 진행 중", unit="파일"):
    file_path = os.path.join(video_folder, file)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"삭제 실패: {file}, 오류: {e}")

print("non_existing_videos 삭제 완료")

