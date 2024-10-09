import torch
import numpy as np
from scipy import signal
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
import json
import torchvision.transforms as transforms
import os, sys, time
from torch.utils.data import DataLoader, ConcatDataset
from data.common_utils import parse_csv_to_list, extract_first_timestamp, remove_timestamp, clean_str
from transformers import PegasusTokenizer



"""
Chapter title generation for single video (specify vid and chapter cut points)
"""
class InferSingleVideoChapterTitleDataset:
    def __init__(self, data_file, vid_file, tokenizer, max_text_len=512, transform=None, target_transform=None):
        """
        Sample a positive or negative clip from dataset  
        Note that the video frame is sampled by 1 frame/second
        """
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)
        self.vid2title = dict()
        self.vid2timestamps = dict()
        self.vid2durations = dict()
        for i in range(len(all_vids)):
            vid = all_vids[i]
            self.vid2title[vid] = titles[i]
            self.vid2timestamps[vid] = timestamps[i]
            self.vid2durations[vid] = durations[i]

        with open(vid_file, "r") as f:
            vids = f.readlines()
            vids = [x.strip() for x in vids]
        self.vids = vids

        # get asr file
        asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
        self.vid2asr_files = dict()
        for asr_file in asr_file_list:
            filename = os.path.basename(asr_file)
            vid = filename.split(".")[0][9:]
            self.vid2asr_files[vid] = asr_file

        self.transform = transform
        self.target_transform = target_transform
    
    """
    you should run choose_vid_and_cut_points before iterate this dataset
    """
    def manual_choose_vid_and_cut_points(self, vid, cut_points):
        if vid in self.vids:
            self.infer_vid = vid
        else:
            raise RuntimeError(f"The vid {vid} is not existed in dataset")
        self._load_gt_data()
        self.cut_points = cut_points

    def _load_gt_data(self):
        vid = self.infer_vid
        timestamp = self.vid2timestamps[vid]
        asr_file = self.vid2asr_files[vid]

        with open(asr_file, "r") as f:
            self.subtitles = json.load(f)

        # parse timestamp
        self.gt_cut_points = []         # G.T. cut point + a little bit offset
        self.gt_real_cut_points = []    # record G.T. cut point
        self.gt_descriptions = []
        for timestamp_str in timestamp:
            gap = 0
            sec, description = extract_first_timestamp(timestamp_str)
            
            for sec_i in range(sec - gap, sec + gap + 1):
                self.gt_cut_points.append(sec_i)
                self.gt_real_cut_points.append(sec)
            self.gt_descriptions.append(description)

    def __len__(self):
        return len(self.cut_points) + 1

    def __getitem__(self, i):
        vid = self.infer_vid
        duration = self.vid2durations[vid]

        if i == 0:
            chapter_start_t = 0
            chapter_end_t = self.cut_points[i]
        elif i == len(self.cut_points):
            chapter_start_t = self.cut_points[i-1]
            chapter_end_t = duration
        else:
            chapter_start_t = self.cut_points[i-1]
            chapter_end_t = self.cut_points[i]
        
        asr_file = self.vid2asr_files[vid]
        with open(asr_file, "r") as f:
            subtitle = json.load(f)

        # get subtitle within selected chapter
        text_extra_time_gap = 1
        text_within_chapter = ""
        for sub in subtitle:
            text = sub["text"]
            start = sub["start"]

            if chapter_start_t - text_extra_time_gap < start < chapter_end_t + text_extra_time_gap:
                if text_within_chapter == "":
                    text_within_chapter = text
                else:
                    text_within_chapter += " " + text

            if start >= chapter_end_t + text_extra_time_gap:
                break
        
        token_list = text_within_chapter.split(" ")
        text_within_chapter = " ".join(token_list)
        text_within_chapter = text_within_chapter.lower()

        # process input text (truncate and pad)
        tokens = self.tokenizer.tokenize(text_within_chapter)
        tokens = tokens[:self.max_text_len]
        attention_mask = [1] * len(tokens)
        if len(tokens) < self.max_text_len:
            zero_pad_list = [0] * (self.max_text_len - len(tokens))
            pad_list = ["<pad>"] * (self.max_text_len - len(tokens))
            tokens += pad_list
            attention_mask += zero_pad_list


        # Convert token to vocabulary indices
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        text_ids = torch.from_numpy(np.array(ids)).long()
        attention_mask = torch.from_numpy(np.array(attention_mask)).long()

        return text_ids, attention_mask




if __name__ == "__main__":
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"
    
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
    dataset = InferSingleVideoChapterTitleDataset(data_file, test_vid_file, tokenizer, max_text_len=512)
    data_loader_params = dict(batch_size=1, shuffle=False, pin_memory=True)
    data_loader = DataLoader(dataset, **data_loader_params)


    dataset.manual_choose_vid_and_cut_points(vid="Nohke4UXGIM", cut_points=[20, 100, 200])
    for text_ids, attention_mask in data_loader:
        print(text_ids.size())
        original_text = tokenizer.batch_decode(text_ids)
        print(original_text)
        print()
