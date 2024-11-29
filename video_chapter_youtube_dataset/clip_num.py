import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# 데이터 로드
df = pd.read_csv('dataset/all_in_one_with_subtitle_final.csv')

# train, validation, test 비디오 ID 로드
def load_video_ids(filepath):
   with open(filepath, 'r') as f:
       return [line.strip() for line in f.readlines()]

train_vids = load_video_ids('dataset/final_train.txt')
val_vids = load_video_ids('dataset/final_validation.txt')
test_vids = load_video_ids('dataset/final_test.txt')

# clip 수 계산 함수
def calculate_num_clips(duration, frame_num=16, max_offset=2):
   sampling_interval = 4 * max_offset  # 4*max_offset 간격으로 sampling
   return int((duration - frame_num) / (sampling_interval)) + 1

# 각 split별 clip 수 계산
def get_clip_counts(video_ids, df):
   durations = df[df['videoId'].isin(video_ids)]['duration']
   return [calculate_num_clips(d) for d in durations]

train_clips = get_clip_counts(train_vids, df)
val_clips = get_clip_counts(val_vids, df)
test_clips = get_clip_counts(test_vids, df)

# 통계 정보 출력
def print_statistics(clips, split_name):
   print(f"\n{split_name} Statistics:")
   print(f"Mean: {np.mean(clips):.2f}")
   print(f"Median: {np.median(clips):.2f}")
   print(f"Min: {np.min(clips)}")
   print(f"Max: {np.max(clips)}")
#    print(f"Total videos: {len(clips)}")
#    print(f"Total clips: {sum(clips)}")

print_statistics(train_clips, "Train")
print_statistics(val_clips, "Validation")
print_statistics(test_clips, "Test")

# 히스토그램 그리기
plt.style.use('seaborn')
splits = [('Train', train_clips), ('Validation', val_clips), ('Test', test_clips)]



for split_name, clips in tqdm(splits, desc='creating plots'):
   plt.figure(figsize=(10, 6))
   plt.hist(clips, bins=50, alpha=0.75, edgecolor='black')
   plt.title(f'Distribution of Clip Counts - {split_name}')
   plt.xlabel('Number of Clips')
   plt.ylabel('Number of Videos')
   
   # 평균값 표시
   mean_clips = np.mean(clips)
   median_clips = np.median(clips)
#    plt.axvline(mean_clips, color='r', linestyle='dashed', linewidth=1)
#    plt.axvline(median_clips, color='g', linestyle='dashed', linewidth=1)
   
   # 범례 추가
#    plt.legend(['Mean', 'Median', 'Distribution'])
   
   # 통계 정보 텍스트 추가
   stats_text = f'Mean: {mean_clips:.1f}\nMedian: {median_clips:.1f}\nMin: {min(clips)}\nMax: {max(clips)}\nTotal Videos: {len(clips)}'
   plt.text(0.95, 0.95, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
   
   plt.tight_layout()
   plt.savefig(f'dataset_stats_result/hist/clip_distribution_{split_name.lower()}.png')
   plt.close()