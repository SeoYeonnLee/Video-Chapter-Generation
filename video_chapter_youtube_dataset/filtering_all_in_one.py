import pandas as pd
import json

csv_file_path = 'dataset/all_in_one_with_subtitle_new.csv'
df_csv = pd.read_csv(csv_file_path)

json_file_path = 'dataset/sampled_videos.json'
with open(json_file_path, 'r') as file:
    video_ids = json.load(file)

video_ids_list = [item for sublist in video_ids.values() for item in sublist]

filtered_df = df_csv[df_csv['videoId'].isin(video_ids_list)]

output_csv_file_path = 'dataset/all_in_one_with_subtitle_final.csv'
filtered_df.to_csv(output_csv_file_path, index=False)

print('Save the filtered DataFrame to a new CSV file')
print(f'new csv file length: {len(filtered_df)}')