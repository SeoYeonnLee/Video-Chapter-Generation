import json
import random

# Set random seed for reproducibility
random.seed(42)

# Path configuration
input_txt = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_validation.txt"
input_json = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/validation_clips_clip_frame_num_16.json"

output_txt = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_validation_50p.txt"
output_json = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/validation_clips_clip_frame_num_16_50p.json"

# Read original validation video IDs
with open(input_txt, 'r') as f:
    video_ids = [line.strip() for line in f.readlines()]

# Sample 60% of video IDs
sample_size = int(len(video_ids) * 0.5)
sampled_video_ids = set(random.sample(video_ids, sample_size))

# Save sampled video IDs to new txt file
with open(output_txt, 'w') as f:
    for vid in sorted(sampled_video_ids):
        f.write(f"{vid}\n")

# Read original json data
with open(input_json, 'r') as f:
    clip_data = json.load(f)

# Filter clips for sampled videos
sampled_clips = [clip for clip in clip_data if clip['vid'] in sampled_video_ids]

# Save filtered clips to new json file
with open(output_json, 'w') as f:
    json.dump(sampled_clips, f)

# Print statistics
print(f"Original number of videos: {len(video_ids)}")
print(f"Sampled number of videos: {len(sampled_video_ids)}")
print(f"Original number of clips: {len(clip_data)}")
print(f"Sampled number of clips: {len(sampled_clips)}")