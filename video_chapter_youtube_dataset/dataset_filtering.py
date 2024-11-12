import os
import json
import random
import numpy as np
from tqdm import tqdm
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list, extract_first_timestamp, clean_str

class DatasetSampler:
    def __init__(self, category_file, target_stats, error_range=0.01):
        
        self.target_stats = target_stats # 목표 통계값들 (논문에서 제공된 값)
        self.error_range = error_range # 허용 오차 범위 (기본값 10%)
        
        with open(os.path.join(category_file), 'r') as f:
            self.category2vid = json.load(f) # 카테고리별 비디오 ID
            
        self.sampled_videos = {}  # 샘플링된 비디오 ID 저장
        self.sampled_stats = {}   # 샘플링된 비디오들의 통계 저장

    # 샘플링된 통계값이 목표 범위 안에 있는지 확인 
    def check_stats_in_range(self, sampled_stats, target_stats, error_range=0.05):

        for stat_name, target_value in target_stats.items():
            if stat_name == 'video_count':
                continue
                
            sampled_value = sampled_stats[stat_name]
            error = abs(sampled_value - target_value) / target_value
            
            if error > error_range:
                return False
        return True
    
    # 비디오들 통계값 계산
    def calculate_stats_for_videos(self, video_ids, category):

        data_file = 'dataset/all_in_one_with_subtitle_new.csv'
        all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)

        category_durations = []
        chapter_durations = []
        category_chapter_nums = []
        category_chapter_word_nums = []

        for vid in tqdm(video_ids, desc=f"Processing Videos in {category}", unit="video", leave=False):
            idx = all_vids.index(vid)

            title = titles[idx]
            duration = durations[idx]
            timestamp = timestamps[idx]

            chapter_num = len(timestamp)

            sorted_timestamps = []
            for line in timestamp:
                sec, description = extract_first_timestamp(line)
                sorted_timestamps.append(sec)
            
            sorted_timestamps.sort()  # 시간순 정렬
            
            # 각 챕터의 duration 계산
            for i in range(len(sorted_timestamps)):
                if i == len(sorted_timestamps) - 1:
                    # 마지막 챕터는 비디오 끝까지
                    chapter_dur = duration - sorted_timestamps[i]
                else:
                    # 다음 챕터 시작까지
                    chapter_dur = sorted_timestamps[i+1] - sorted_timestamps[i]
                chapter_durations.append(chapter_dur)

            chapter_word_num = 0
            for line in timestamp:
                sec, description = extract_first_timestamp(line)
                description = clean_str(description)
                chapter_word_num += len(description.split(" "))

            category_durations.append(duration)
            category_chapter_nums.append(chapter_num)
            category_chapter_word_nums.append(chapter_word_num)

        stats = {
            "video_count": len(video_ids),
            "avg_chapter_duration": round(sum(category_durations) / sum(category_chapter_nums), 2), 
            "avg_chapters_per_video": round(sum(category_chapter_nums) / len(video_ids), 2), 
            "avg_words_per_chapter": round(sum(category_chapter_word_nums) / sum(category_chapter_nums), 2)
        }

        return stats
    
    # 한 카테고리에 대해 샘플링
    def sample_category(self, category):

        target = self.target_stats[category]
        available_videos = self.category2vid[category]  # 해당 카테고리의 모든 비디오

        if category == "Category:Youth":
            sampled_videos = available_videos
            sampled_stats = self.calculate_stats_for_videos(available_videos, category)
            self.sampled_videos[category] = sampled_videos
            self.sampled_stats[category] = sampled_stats
            print(f"{category} sampled successfully")
            return True
        
        # 목표 비디오 수가 가용 비디오 수보다 많은 경우 체크
        if target['video_count'] > len(available_videos):
            print(f"Warning: Requested {target['video_count']} videos for {category} "
                  f"but only {len(available_videos)} available")
            return False
        
        max_attempts = 500
        attempt = 0
        
        while attempt < max_attempts:
            # 비복원추출로 샘플링
            sampled_videos = random.sample(available_videos, target['video_count'])
            
            # 통계 계산
            sampled_stats = self.calculate_stats_for_videos(sampled_videos, category)
            
            # 통계값이 범위 안에 들어오는지 확인
            if self.check_stats_in_range(sampled_stats, target, 0.05):
                self.sampled_videos[category] = sampled_videos
                self.sampled_stats[category] = sampled_stats
                print(f"{category} sampled successfully after {attempt+1} attempts")
                return True
                
            attempt += 1
        
        while attempt < (2 * max_attempts):
            # 비복원추출로 샘플링
            sampled_videos = random.sample(available_videos, target['video_count'])
            
            # 통계 계산
            sampled_stats = self.calculate_stats_for_videos(sampled_videos, category)
            
            # 통계값이 범위 안에 들어오는지 확인
            if self.check_stats_in_range(sampled_stats, target, 0.1):
                self.sampled_videos[category] = sampled_videos
                self.sampled_stats[category] = sampled_stats
                print(f"{category} sampled successfully after {attempt+1} attempts")
                return True
                
            attempt += 1
            
        print(f"Failed to sample {category} after {max_attempts} attempts")
        return False
    
    # 모든 카테고리에 대해 샘플링 수행
    def sample_all_categories(self):

        success_count = 0
        for category in tqdm(self.target_stats.keys(), desc="Sampling categories"):
            if self.sample_category(category):
                success_count += 1
                
        print(f"\nSuccessfully sampled {success_count}/{len(self.target_stats)} categories")
    
    # 샘플링 결과 저장
    def save_results(self, video_file, stats_file):

        # 샘플링된 비디오 ID 저장
        with open(video_file, 'w') as f:
            json.dump(self.sampled_videos, f, indent=4)
            
        # 통계값 저장
        with open(stats_file, 'w') as f:
            json.dump(self.sampled_stats, f, indent=4)
            
def main():
    # 목표 통계값
    with open('dataset_stats_result/target_stats.json', 'r') as f:
        target_stats = json.load(f)
    
    # 샘플러 초기화
    sampler = DatasetSampler(
        category_file="category2total_vid_valid.json",
        target_stats=target_stats,
        error_range=0.05
    )
    
    # 샘플링 수행
    sampler.sample_all_categories()
    
    # 결과 저장
    sampler.save_results(
        video_file="dataset/sampled_videos.json",
        stats_file="dataset/sampled_statistics.json"
    )

if __name__ == "__main__":
    main()