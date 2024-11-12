import os
import json
import glob
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list, extract_first_timestamp, clean_str
from tqdm import tqdm

class InvalidVideoIDExtractor:
    def __init__(self, img_dir, data_file, category_vid_file):
        self.img_dir = img_dir
        self.invalid_vids = set()

        # 데이터 파일에서 비디오 ID와 타임스탬프 매핑
        all_vids, _, durations, timestamps = parse_csv_to_list(data_file)
        self.vid2timestamps = {all_vids[i]: timestamps[i] for i in range(len(all_vids))}
        self.vid2durations = {all_vids[i]: durations[i] for i in range(len(all_vids))}

        # 카테고리별 비디오 ID 로드
        with open(category_vid_file, "r") as f:
            self.category_vids = json.load(f)

        # 자막 파일 경로 매핑
        asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
        self.vid2asr_files = {os.path.basename(f).split(".")[0][9:]: f for f in asr_file_list}

    def extract_invalid_vids(self):
        """비디오 ID를 검증하여 유효하지 않은 ID를 찾음"""
        for category, vids in tqdm(self.category_vids.items(), desc="Processing categories", unit="category"):
            for vid in tqdm(vids, desc=f"Processing category: {category}", unit="video"):
                timestamp = self.vid2timestamps.get(vid, [])
                asr_file = self.vid2asr_files.get(vid, None)
                if not asr_file:
                    continue

                image_path = os.path.join(self.img_dir, vid)
                image_num = len(glob.glob(image_path + "/*.jpg"))

                for timestamp_str in timestamp:
                    sec, _ = extract_first_timestamp(timestamp_str)
                    if sec > image_num:
                        self.invalid_vids.add(vid)
                        break

            # 유효하지 않은 ID 제거
            self.category_vids[category] = [vid for vid in vids if vid not in self.invalid_vids]

    def save_valid_vids(self, filepath="category2total_vid_valid.json"):
        """유효한 비디오 ID를 JSON 파일로 저장"""
        with open(filepath, "w") as f:
            json.dump(self.category_vids, f, indent=4)

    def save_invalid_vids(self, filepath="invalid_vids_total.txt"):
        """유효하지 않은 비디오 ID를 텍스트 파일로 저장"""
        with open(filepath, "w") as f:
            for vid in self.invalid_vids:
                f.write(f"{vid}\n")

if __name__ == "__main__":
    # 경로 설정 예시
    img_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/youtube_video_frame_dataset"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_new.csv"
    category_vid_file = "dataset/category2total_vid.json"

    extractor = InvalidVideoIDExtractor(img_dir, data_file, category_vid_file)
    extractor.extract_invalid_vids()
    extractor.save_valid_vids("category2total_vid_valid.json")
    extractor.save_invalid_vids("dataset/invalid_vids_total.txt")
