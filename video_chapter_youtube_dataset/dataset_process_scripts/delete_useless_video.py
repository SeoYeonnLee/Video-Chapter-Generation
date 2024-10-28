"""
This script will delete downloaded videos to save disk space.
WARNING: use this script after you have prepared all data.

"""


import os, glob
import pandas as pd


data = pd.read_csv("/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_new.csv")
vids = list(data["videoId"].values)
durations = list(data["duration"].values)
video_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/vids"
video_files = [video_dir + "/" + x + ".mp4" for x in vids]
video_filenames = set([os.path.basename(x) for x in video_files])

all_files = glob.glob(video_dir + "/*")
all_video_files = glob.glob(video_dir + "/*.mp4")

other_files = set(all_files) - set(all_video_files)
for filename in other_files:
    os.remove(filename)

all_video_filenames = set([os.path.basename(x) for x in all_video_files])

overlap_video_filenames = all_video_filenames.intersection(video_filenames)
diff_video_filenames = all_video_filenames.difference(video_filenames)

assert len(overlap_video_filenames) == len(video_filenames)

print(f"total {len(diff_video_filenames)} videos need to be deleted")
for i, filename in enumerate(diff_video_filenames):
    print(f"removing {i}/{len(diff_video_filenames)}, {filename}")
    os.remove(os.path.join(video_dir, filename))



