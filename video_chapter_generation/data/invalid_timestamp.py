import os
import json
import glob
from common_utils import parse_csv_to_list, extract_first_timestamp
from tqdm import tqdm

class InvalidVideoIDExtractor:
    def __init__(self, img_dir, data_file, vid_file):
        self.img_dir = img_dir
        self.invalid_vids = set()  # 중복 제거를 위해 set 사용

        all_vids, _, durations, timestamps = parse_csv_to_list(data_file)
        self.vid2timestamps = {all_vids[i]: timestamps[i] for i in range(len(all_vids))}
        self.vid2durations = {all_vids[i]: durations[i] for i in range(len(all_vids))}

        with open(vid_file, "r") as f:
            self.vids = [x.strip() for x in f.readlines()]

        asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
        self.vid2asr_files = {os.path.basename(f).split(".")[0][9:]: f for f in asr_file_list}

    def extract_invalid_vids(self):
        """조건에 맞는 비디오 ID를 찾고 set에 추가"""
        for vid in tqdm(self.vids, desc="Processing videos", unit="video"):
            timestamp = self.vid2timestamps[vid]
            asr_file = self.vid2asr_files.get(vid, None)
            if not asr_file:
                continue

            image_path = os.path.join(self.img_dir, vid)
            image_num = len(glob.glob(image_path + "/*.jpg"))

            for timestamp_str in timestamp:
                sec, _ = extract_first_timestamp(timestamp_str)
                if sec > image_num:  # 조건에 맞는 경우 비디오 ID 추가
                    self.invalid_vids.add(vid)
                    break

    def save_invalid_vids(self, filepath="invalid_vids.txt"):
        """조건에 맞는 비디오 ID를 텍스트 파일로 저장"""
        with open(filepath, "w") as f:
            for vid in self.invalid_vids:
                f.write(f"{vid}\n")


if __name__ == "__main__":
    # 예시 경로 설정
    img_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/youtube_video_frame_dataset"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_new.csv"
    vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/train_final.txt"

    extractor = InvalidVideoIDExtractor(img_dir, data_file, vid_file)
    extractor.extract_invalid_vids()
    extractor.save_invalid_vids("invalid_vids.txt")
