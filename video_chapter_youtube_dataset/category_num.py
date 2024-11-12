import json

def save_category_video_count(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    category_video_count = {category: len(videos) for category, videos in data.items()}
    
    with open(output_file, 'w') as f:
        json.dump(category_video_count, f, indent=4)

if __name__ == "__main__":
    input_file = 'category2total_vid_valid.json'
    output_file = 'dataset_stats_result/valid_timestamp_video_num.json'
    save_category_video_count(input_file, output_file)