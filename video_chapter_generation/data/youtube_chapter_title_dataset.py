"""
a dataset for training and testing chapter title generation
"""


import json
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


# sample a chapter for each video
class YoutubeChapterTitleDataset:
    def __init__(self, data_file, vid_file, tokenizer, max_text_len=512, chapter_title_text_len=30, transform=None, target_transform=None):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.chapter_title_text_len = chapter_title_text_len
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

    def __getitem__(self, i):
        vid = self.vids[i]
        # print(vid)
        timestamp = self.vid2timestamps[vid]
        asr_file = self.vid2asr_files[vid]
        duration = round(self.vid2durations[vid] - 1)    # equal to video total duration (seconds)

        with open(asr_file, "r") as f:
            subtitle = json.load(f)

        # extract timestamp
        timepoint_secs = []
        descriptions = []
        for line in timestamp:
            sec, description = extract_first_timestamp(line)
            timepoint_secs.append(sec)
            descriptions.append(description)

        # randomly select a chapter
        chapter_idx = random.randint(0, len(descriptions) - 1)
        description = descriptions[chapter_idx]
        description = clean_str(description)
        description = remove_timestamp(description)
        description = description.lower()
        chapter_start_t = timepoint_secs[chapter_idx]
        if chapter_idx + 1 < len(timepoint_secs):
            chapter_end_t = timepoint_secs[chapter_idx + 1]
        else:
            chapter_end_t = duration

        # get subtitle within selected chapter
        time_gap = 1
        text_within_chapter = ""
        for sub in subtitle:
            text = sub["text"]
            start = sub["start"]

            if chapter_start_t - time_gap < start < chapter_end_t + time_gap:
                if text_within_chapter == "":
                    text_within_chapter = text
                else:
                    text_within_chapter += " " + text

            if start >= chapter_end_t + time_gap:
                break
        
        token_list = text_within_chapter.split(" ")
        text_within_chapter = " ".join(token_list)
        text_within_chapter = text_within_chapter.lower()

        # process input text (truncate and pad)
        pad_token = self.tokenizer.pad_token

        tokens = self.tokenizer.tokenize(text_within_chapter)
        tokens = tokens[:self.max_text_len]
        attention_mask = [1] * len(tokens)
        if len(tokens) < self.max_text_len:
            zero_pad_list = [0] * (self.max_text_len - len(tokens))
            pad_list = [pad_token] * (self.max_text_len - len(tokens))
            tokens += pad_list
            attention_mask += zero_pad_list

        # process summarization text (truncate and pad and shift right to make label)
        # bos_token = self.tokenizer.bos_token
        # if bos_token is None:       # google/pegasus-large has no bos_token, but model config use pad token as decoder_start_token_id
        #     bos_token = pad_token
        bos_token = pad_token

        decode_tokens = self.tokenizer.tokenize(description)
        input_decode_tokens = [bos_token] + decode_tokens   # bos_token is decoder_start_token
        input_decode_tokens = input_decode_tokens[:self.chapter_title_text_len]

        eos_token = self.tokenizer.eos_token
        if len(decode_tokens) >= self.chapter_title_text_len:
            target_decode_tokens = decode_tokens
            target_decode_tokens[self.chapter_title_text_len - 1] = eos_token
        else:
            target_decode_tokens = decode_tokens + [eos_token]
        target_decode_tokens = target_decode_tokens[:self.chapter_title_text_len]
        
        decode_attention_mask = [1] * (len(decode_tokens) + 1)
        decode_attention_mask = decode_attention_mask[:self.chapter_title_text_len]
        if len(decode_attention_mask) < self.chapter_title_text_len:
            zero_pad_list = [0] * (self.chapter_title_text_len - len(decode_attention_mask))
            pad_list = [eos_token] * (self.chapter_title_text_len - len(decode_attention_mask))
            input_decode_tokens += pad_list
            target_decode_tokens += pad_list
            decode_attention_mask += zero_pad_list

        # Convert token to vocabulary indices
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        text_ids = torch.from_numpy(np.array(ids)).long()
        attention_mask = torch.from_numpy(np.array(attention_mask)).long()

        input_decode_ids = self.tokenizer.convert_tokens_to_ids(input_decode_tokens)
        input_decode_ids = torch.from_numpy(np.array(input_decode_ids)).long()
        decode_attention_mask = torch.from_numpy(np.array(decode_attention_mask)).long()
        target_decode_ids = self.tokenizer.convert_tokens_to_ids(target_decode_tokens)
        target_decode_ids = torch.from_numpy(np.array(target_decode_ids)).long()

        return text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids

    def __len__(self):
        return len(self.vids)


# title dataset with subtitle and images
class YoutubeChapterTitleWithVisionEmbDataset:
    def __init__(self, vision_emb_dir, data_file, vid_file, tokenizer, max_text_len=512, chapter_title_text_len=30, max_vision_emb=10):
        self.vision_emb_dir = vision_emb_dir
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.chapter_title_text_len = chapter_title_text_len
        self.max_vision_emb = max_vision_emb
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

    def __getitem__(self, i):
        vid = self.vids[i]
        # print(vid)
        timestamp = self.vid2timestamps[vid]
        asr_file = self.vid2asr_files[vid]
        duration = round(self.vid2durations[vid] - 1)    # equal to video total duration (seconds)

        with open(asr_file, "r") as f:
            subtitle = json.load(f)

        # extract timestamp
        timepoint_secs = []
        descriptions = []
        for line in timestamp:
            sec, description = extract_first_timestamp(line)
            timepoint_secs.append(sec)
            descriptions.append(description)

        # randomly select a chapter
        chapter_idx = random.randint(0, len(descriptions) - 1)
        description = descriptions[chapter_idx]
        description = clean_str(description)
        description = remove_timestamp(description)
        description = description.lower()
        chapter_start_t = timepoint_secs[chapter_idx]
        if chapter_idx + 1 < len(timepoint_secs):
            chapter_end_t = timepoint_secs[chapter_idx + 1]
        else:
            chapter_end_t = duration

        # get corresponding vision embs within a chapter segment
        vision_emb_dir = os.path.join(self.vision_emb_dir, vid)
        emb_start_idx = int(chapter_start_t // 4) * 4
        emb_end_idx = int(chapter_end_t // 4) * 4 - 16
        if emb_end_idx < 0:
            emb_end_idx = emb_start_idx
        if emb_start_idx > emb_end_idx:
            emb_start_idx = emb_end_idx
        
        vision_embs = []
        for st in range(emb_start_idx, emb_end_idx+1, 16):
            emb_path = os.path.join(vision_emb_dir, f"vision_emb_{st}_{st+16}.npy")
            with open(emb_path, 'rb') as f:
                emb = np.load(f)
                emb = np.mean(emb, axis=0)
            vision_embs.append(emb)
        vision_embs = vision_embs[:self.max_vision_emb]
        vision_attention_mask = [1] * len(vision_embs)
        if len(vision_embs) < self.max_vision_emb:
            pad_size = self.max_vision_emb - len(vision_embs)
            pad_embs = np.zeros_like(emb)
            vision_embs = vision_embs + [pad_embs] * pad_size
            vision_attention_mask = vision_attention_mask + [0] * pad_size
        vision_embs = np.stack(vision_embs, axis=0)

        vision_embs = torch.from_numpy(vision_embs).float()
        vision_attention_mask = torch.from_numpy(np.array(vision_attention_mask)).long()


        # get subtitle within selected chapter
        time_gap = 1
        text_within_chapter = ""
        for sub in subtitle:
            text = sub["text"]
            start = sub["start"]

            if chapter_start_t - time_gap < start < chapter_end_t + time_gap:
                if text_within_chapter == "":
                    text_within_chapter = text
                else:
                    text_within_chapter += " " + text

            if start >= chapter_end_t + time_gap:
                break
        
        token_list = text_within_chapter.split(" ")
        text_within_chapter = " ".join(token_list)
        text_within_chapter = text_within_chapter.lower()

        # process input text (truncate and pad)
        pad_token = self.tokenizer.pad_token

        tokens = self.tokenizer.tokenize(text_within_chapter)
        tokens = tokens[:self.max_text_len]
        attention_mask = [1] * len(tokens)
        if len(tokens) < self.max_text_len:
            zero_pad_list = [0] * (self.max_text_len - len(tokens))
            pad_list = [pad_token] * (self.max_text_len - len(tokens))
            tokens += pad_list
            attention_mask += zero_pad_list

        # process summarization text (truncate and pad and shift right to make label)
        # bos_token = self.tokenizer.bos_token
        # if bos_token is None:       # google/pegasus-large has no bos_token, but model config use pad token as decoder_start_token_id
        #     bos_token = pad_token
        bos_token = pad_token

        decode_tokens = self.tokenizer.tokenize(description)
        input_decode_tokens = [bos_token] + decode_tokens   # bos_token is decoder_start_token
        input_decode_tokens = input_decode_tokens[:self.chapter_title_text_len]

        eos_token = self.tokenizer.eos_token
        if len(decode_tokens) >= self.chapter_title_text_len:
            target_decode_tokens = decode_tokens
            target_decode_tokens[self.chapter_title_text_len - 1] = eos_token
        else:
            target_decode_tokens = decode_tokens + [eos_token]
        target_decode_tokens = target_decode_tokens[:self.chapter_title_text_len]
        
        decode_attention_mask = [1] * (len(decode_tokens) + 1)
        decode_attention_mask = decode_attention_mask[:self.chapter_title_text_len]
        if len(decode_attention_mask) < self.chapter_title_text_len:
            zero_pad_list = [0] * (self.chapter_title_text_len - len(decode_attention_mask))
            pad_list = [eos_token] * (self.chapter_title_text_len - len(decode_attention_mask))
            input_decode_tokens += pad_list
            target_decode_tokens += pad_list
            decode_attention_mask += zero_pad_list

        # Convert token to vocabulary indices
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        text_ids = torch.from_numpy(np.array(ids)).long()
        attention_mask = torch.from_numpy(np.array(attention_mask)).long()

        input_decode_ids = self.tokenizer.convert_tokens_to_ids(input_decode_tokens)
        input_decode_ids = torch.from_numpy(np.array(input_decode_ids)).long()
        decode_attention_mask = torch.from_numpy(np.array(decode_attention_mask)).long()
        target_decode_ids = self.tokenizer.convert_tokens_to_ids(target_decode_tokens)
        target_decode_ids = torch.from_numpy(np.array(target_decode_ids)).long()

        return vision_embs, vision_attention_mask, text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids

    def __len__(self):
        return len(self.vids)




# flat all chapters in dataset and traverse all chapters
class YoutubeAllChapterTitleDataset:
    def __init__(self, data_file, vid_file, tokenizer, max_text_len=512, chapter_title_text_len=30, vision_emb_dir=None, max_vision_emb=10):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.chapter_title_text_len = chapter_title_text_len
        self.vision_emb_dir = vision_emb_dir
        self.max_vision_emb = max_vision_emb
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

        # get all text by chapter within dataset
        self.all_data = []
        for vid in self.vids:
            chapter_text_data_list = []
            timestamp = self.vid2timestamps[vid]
            asr_file = self.vid2asr_files[vid]
            duration = round(self.vid2durations[vid] - 1)    # equal to video total duration (seconds)

            with open(asr_file, "r") as f:
                subtitle = json.load(f)
            
            # extract timestamp
            timepoint_secs = []
            descriptions = []
            for line in timestamp:
                sec, description = extract_first_timestamp(line)
                timepoint_secs.append(sec)
                descriptions.append(description)
            
            # tranverse all chapters in this vid
            for chapter_idx in range(len(descriptions)):
                description = descriptions[chapter_idx]
                description = clean_str(description)
                description = remove_timestamp(description)
                description = description.lower()
                chapter_start_t = timepoint_secs[chapter_idx]
                if chapter_idx + 1 < len(timepoint_secs):
                    chapter_end_t = timepoint_secs[chapter_idx + 1]
                else:
                    chapter_end_t = duration
                
                # get subtitle within selected chapter
                time_gap = 2
                text_within_chapter = ""
                for sub in subtitle:
                    text = sub["text"]
                    start = sub["start"]

                    if chapter_start_t - time_gap < start < chapter_end_t + time_gap:
                        if text_within_chapter == "":
                            text_within_chapter = text
                        else:
                            text_within_chapter += " " + text

                    if start >= chapter_end_t + time_gap:
                        break
                
                chapter_text_data = {
                    "vid": vid,
                    "chapter_start_t": chapter_start_t,
                    "chapter_end_t": chapter_end_t,
                    "text_within_chapter": text_within_chapter,
                    "description": description
                }
                chapter_text_data_list.append(chapter_text_data)

            self.all_data.extend(chapter_text_data_list)


    def __getitem__(self, i):
        vid = self.all_data[i]["vid"]
        chapter_start_t = self.all_data[i]["chapter_start_t"]
        chapter_end_t = self.all_data[i]["chapter_end_t"]

        if self.vision_emb_dir:
            vision_emb_dir = os.path.join(self.vision_emb_dir, vid)
            emb_start_idx = int(chapter_start_t // 4) * 4
            emb_end_idx = int(chapter_end_t // 4) * 4 - 16
            if emb_end_idx < 0:
                emb_end_idx = emb_start_idx
            if emb_start_idx > emb_end_idx:
                emb_start_idx = emb_end_idx
            
            vision_embs = []
            for st in range(emb_start_idx, emb_end_idx+1, 16):
                emb_path = os.path.join(vision_emb_dir, f"vision_emb_{st}_{st+16}.npy")
                with open(emb_path, 'rb') as f:
                    emb = np.load(f)
                    emb = np.mean(emb, axis=0)
                vision_embs.append(emb)
            vision_embs = vision_embs[:self.max_vision_emb]
            vision_attention_mask = [1] * len(vision_embs)
            if len(vision_embs) < self.max_vision_emb:
                pad_size = self.max_vision_emb - len(vision_embs)
                pad_embs = np.zeros_like(emb)
                vision_embs = vision_embs + [pad_embs] * pad_size
                vision_attention_mask = vision_attention_mask + [0] * pad_size
            vision_embs = np.stack(vision_embs, axis=0)

            vision_embs = torch.from_numpy(vision_embs).float()
            vision_attention_mask = torch.from_numpy(np.array(vision_attention_mask)).long()



        text_within_chapter = self.all_data[i]["text_within_chapter"]
        description = self.all_data[i]["description"]
        
        token_list = text_within_chapter.split(" ")
        text_within_chapter = " ".join(token_list)
        text_within_chapter = text_within_chapter.lower()

        # process input text (truncate and pad)
        pad_token = self.tokenizer.pad_token

        tokens = self.tokenizer.tokenize(text_within_chapter)
        tokens = tokens[:self.max_text_len]
        attention_mask = [1] * len(tokens)
        if len(tokens) < self.max_text_len:
            zero_pad_list = [0] * (self.max_text_len - len(tokens))
            pad_list = [pad_token] * (self.max_text_len - len(tokens))
            tokens += pad_list
            attention_mask += zero_pad_list

        # process summarization text (truncate and pad and shift right to make label)
        # bos_token = self.tokenizer.bos_token
        # if bos_token is None:       # google/pegasus-large has no bos_token, but model config use pad token as decoder_start_token_id
        #     bos_token = pad_token
        bos_token = pad_token

        decode_tokens = self.tokenizer.tokenize(description)
        input_decode_tokens = [bos_token] + decode_tokens   # bos_token is decoder_start_token
        input_decode_tokens = input_decode_tokens[:self.chapter_title_text_len]

        eos_token = self.tokenizer.eos_token
        if len(decode_tokens) >= self.chapter_title_text_len:
            target_decode_tokens = decode_tokens
            target_decode_tokens[self.chapter_title_text_len - 1] = eos_token
        else:
            target_decode_tokens = decode_tokens + [eos_token]
        target_decode_tokens = target_decode_tokens[:self.chapter_title_text_len]
        
        decode_attention_mask = [1] * (len(decode_tokens) + 1)
        decode_attention_mask = decode_attention_mask[:self.chapter_title_text_len]
        if len(decode_attention_mask) < self.chapter_title_text_len:
            zero_pad_list = [0] * (self.chapter_title_text_len - len(decode_attention_mask))
            pad_list = [eos_token] * (self.chapter_title_text_len - len(decode_attention_mask))
            input_decode_tokens += pad_list
            target_decode_tokens += pad_list
            decode_attention_mask += zero_pad_list

        # Convert token to vocabulary indices
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        text_ids = torch.from_numpy(np.array(ids)).long()
        attention_mask = torch.from_numpy(np.array(attention_mask)).long()

        input_decode_ids = self.tokenizer.convert_tokens_to_ids(input_decode_tokens)
        input_decode_ids = torch.from_numpy(np.array(input_decode_ids)).long()
        decode_attention_mask = torch.from_numpy(np.array(decode_attention_mask)).long()
        target_decode_ids = self.tokenizer.convert_tokens_to_ids(target_decode_tokens)
        target_decode_ids = torch.from_numpy(np.array(target_decode_ids)).long()

        if self.vision_emb_dir:
            return vision_embs, vision_attention_mask, text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids
        
        return text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids

    def __len__(self):
        return len(self.all_data)


# using predict cut points to generate title
class YoutubeAllChapterTitlePredictDataset:
    def __init__(self, vid2cut_points_file, data_file, vid_file, tokenizer, max_text_len=512, chapter_title_text_len=30, vision_emb_dir=None, max_vision_emb=10):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.chapter_title_text_len = chapter_title_text_len
        self.vision_emb_dir = vision_emb_dir
        self.max_vision_emb = max_vision_emb
        all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)
        self.vid2title = dict()
        self.vid2timestamps = dict()
        self.vid2durations = dict()
        for i in range(len(all_vids)):
            vid = all_vids[i]
            self.vid2title[vid] = titles[i]
            self.vid2timestamps[vid] = timestamps[i]
            self.vid2durations[vid] = durations[i]

        with open(vid2cut_points_file, "r") as f:
            vid2cut_points = json.load(f)
        self.vid2pred_cut_points = vid2cut_points

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

        # get all text by chapter within dataset
        self.all_data = []
        for vid in self.vids:
            chapter_text_data_list = []
            timestamp = self.vid2timestamps[vid]
            asr_file = self.vid2asr_files[vid]
            duration = round(self.vid2durations[vid] - 1)    # equal to video total duration (seconds)
            pred_cut_points = self.vid2pred_cut_points[vid]["second_pred_cut_points"]

            with open(asr_file, "r") as f:
                subtitle = json.load(f)
            
            # extract timestamp
            gt_timepoint_secs = []
            descriptions = []
            for line in timestamp:
                sec, description = extract_first_timestamp(line)
                gt_timepoint_secs.append(sec)
                descriptions.append(description)
            
            # find the nearest predicted timepoint to G.T. timepoint
            nearest_pred_cut_points = []
            for t in gt_timepoint_secs:
                nearest_pred_cut_point = min(pred_cut_points, key=lambda x:abs(x-t))
                nearest_pred_cut_points.append(nearest_pred_cut_point)

            
            # tranverse all chapters in this vid
            for chapter_idx in range(len(descriptions)):
                description = descriptions[chapter_idx]
                description = clean_str(description)
                description = remove_timestamp(description)
                description = description.lower()
                chapter_start_t = nearest_pred_cut_points[chapter_idx]
                if chapter_idx + 1 < len(nearest_pred_cut_points):
                    chapter_end_t = nearest_pred_cut_points[chapter_idx + 1]
                else:
                    chapter_end_t = duration
                
                # get subtitle within selected chapter
                time_gap = 2
                text_within_chapter = ""
                for sub in subtitle:
                    text = sub["text"]
                    start = sub["start"]

                    if chapter_start_t - time_gap < start < chapter_end_t + time_gap:
                        if text_within_chapter == "":
                            text_within_chapter = text
                        else:
                            text_within_chapter += " " + text

                    if start >= chapter_end_t + time_gap:
                        break
                
                chapter_text_data = {
                    "vid": vid,
                    "pred_chapter_start_t": chapter_start_t,
                    "pred_chapter_end_t": chapter_end_t,
                    "text_within_chapter": text_within_chapter,
                    "description": description
                }
                chapter_text_data_list.append(chapter_text_data)

            self.all_data.extend(chapter_text_data_list)


    def __getitem__(self, i):
        vid = self.all_data[i]["vid"]
        pred_chapter_start_t = self.all_data[i]["pred_chapter_start_t"]
        pred_chapter_end_t = self.all_data[i]["pred_chapter_end_t"]

        if self.vision_emb_dir:
            vision_emb_dir = os.path.join(self.vision_emb_dir, vid)
            emb_start_idx = int(pred_chapter_start_t // 4) * 4
            emb_end_idx = int(pred_chapter_end_t // 4) * 4 - 16
            if emb_end_idx < 0:
                emb_end_idx = emb_start_idx
            if emb_start_idx > emb_end_idx:
                emb_start_idx = emb_end_idx
            
            vision_embs = []
            for st in range(emb_start_idx, emb_end_idx+1, 16):
                emb_path = os.path.join(vision_emb_dir, f"vision_emb_{st}_{st+16}.npy")
                with open(emb_path, 'rb') as f:
                    emb = np.load(f)
                    emb = np.mean(emb, axis=0)
                vision_embs.append(emb)
            vision_embs = vision_embs[:self.max_vision_emb]
            vision_attention_mask = [1] * len(vision_embs)
            if len(vision_embs) < self.max_vision_emb:
                pad_size = self.max_vision_emb - len(vision_embs)
                pad_embs = np.zeros_like(emb)
                vision_embs = vision_embs + [pad_embs] * pad_size
                vision_attention_mask = vision_attention_mask + [0] * pad_size
            vision_embs = np.stack(vision_embs, axis=0)

            vision_embs = torch.from_numpy(vision_embs).float()
            vision_attention_mask = torch.from_numpy(np.array(vision_attention_mask)).long()


        text_within_chapter = self.all_data[i]["text_within_chapter"]
        description = self.all_data[i]["description"]
        
        token_list = text_within_chapter.split(" ")
        text_within_chapter = " ".join(token_list)
        text_within_chapter = text_within_chapter.lower()

        # process input text (truncate and pad)
        pad_token = self.tokenizer.pad_token

        tokens = self.tokenizer.tokenize(text_within_chapter)
        tokens = tokens[:self.max_text_len]
        attention_mask = [1] * len(tokens)
        if len(tokens) < self.max_text_len:
            zero_pad_list = [0] * (self.max_text_len - len(tokens))
            pad_list = [pad_token] * (self.max_text_len - len(tokens))
            tokens += pad_list
            attention_mask += zero_pad_list

        # process summarization text (truncate and pad and shift right to make label)
        # bos_token = self.tokenizer.bos_token
        # if bos_token is None:       # google/pegasus-large has no bos_token, but model config use pad token as decoder_start_token_id
        #     bos_token = pad_token
        bos_token = pad_token

        decode_tokens = self.tokenizer.tokenize(description)
        input_decode_tokens = [bos_token] + decode_tokens   # bos_token is decoder_start_token
        input_decode_tokens = input_decode_tokens[:self.chapter_title_text_len]

        eos_token = self.tokenizer.eos_token
        if len(decode_tokens) >= self.chapter_title_text_len:
            target_decode_tokens = decode_tokens
            target_decode_tokens[self.chapter_title_text_len - 1] = eos_token
        else:
            target_decode_tokens = decode_tokens + [eos_token]
        target_decode_tokens = target_decode_tokens[:self.chapter_title_text_len]
        
        decode_attention_mask = [1] * (len(decode_tokens) + 1)
        decode_attention_mask = decode_attention_mask[:self.chapter_title_text_len]
        if len(decode_attention_mask) < self.chapter_title_text_len:
            zero_pad_list = [0] * (self.chapter_title_text_len - len(decode_attention_mask))
            pad_list = [eos_token] * (self.chapter_title_text_len - len(decode_attention_mask))
            input_decode_tokens += pad_list
            target_decode_tokens += pad_list
            decode_attention_mask += zero_pad_list

        # Convert token to vocabulary indices
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        text_ids = torch.from_numpy(np.array(ids)).long()
        attention_mask = torch.from_numpy(np.array(attention_mask)).long()

        input_decode_ids = self.tokenizer.convert_tokens_to_ids(input_decode_tokens)
        input_decode_ids = torch.from_numpy(np.array(input_decode_ids)).long()
        decode_attention_mask = torch.from_numpy(np.array(decode_attention_mask)).long()
        target_decode_ids = self.tokenizer.convert_tokens_to_ids(target_decode_tokens)
        target_decode_ids = torch.from_numpy(np.array(target_decode_ids)).long()

        if self.vision_emb_dir:
            return vision_embs, vision_attention_mask, text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids

        return text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids

    def __len__(self):
        return len(self.all_data)



if __name__ == "__main__":
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    vision_emb_dir = "/opt/tiger/youtube_video_vision_emb_clip_frame_num_16"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"
    glove_pickle_file = "/opt/tiger/video_chapter_generation/glove.840B.300d.pickle"
    vid2cut_points_file = "/opt/tiger/video_chapter_generation/test_results/all/two_stream_validation/batch_32_head_type_mlp_clip_frame_num_16_all_vid2cut_points.json"
    
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
    tokenizer2 = PegasusTokenizer.from_pretrained('google/bigbird-pegasus-large-arxiv')
    # dataset = YoutubeChapterTitleDataset(data_file, train_vid_file, tokenizer, max_text_len=512, chapter_title_text_len=30)
    # dataset = YoutubeAllChapterTitleDataset(data_file, test_vid_file, tokenizer, max_text_len=512, chapter_title_text_len=30)
    # dataset = YoutubeAllChapterTitlePredictDataset(vid2cut_points_file, data_file, test_vid_file, tokenizer, max_text_len=512, chapter_title_text_len=30)
    
    # this dataset will produce subtitle and vision emb within a chapter
    dataset = YoutubeChapterTitleWithVisionEmbDataset(vision_emb_dir, data_file, train_vid_file, tokenizer, max_text_len=512, chapter_title_text_len=30, max_vision_emb=10)
    
    # data_loader_params = dict(batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    data_loader_params = dict(batch_size=64, shuffle=False, pin_memory=True)
    data_loader = DataLoader(dataset, **data_loader_params)

    print(f"number of chapters {len(dataset)}")
    # for text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids in data_loader:
    #     print(text_ids.size())
    #     print(input_decode_ids.size())
    #     print()


    for vision_embs, vision_attention_mask, text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids in data_loader:
        print(vision_embs.size())
        print(vision_attention_mask.size())
        print(text_ids.size())
        print(input_decode_ids.size())
        print()
