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
from data.common_utils import parse_csv_to_list, extract_first_timestamp, load_glove_from_pickle, text_decontracted
from transformers import BertTokenizer
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import random


X_PAD = 0
Y_PAD = -1


class YoutubeClipDataset:
    def __init__(self, img_dir, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, mode="all", transform=None, target_transform=None):
        """
        Sample a positive or negative clip from dataset  
        Note that the video frame is sampled by 1 frame/second
        """
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.mode = mode        # text (text only), image (image only) or all (multiple-model)
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        self.img_dir = img_dir
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

        # asr files
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
        timestamp = self.vid2timestamps[vid]
        asr_file = self.vid2asr_files[vid]
        image_path = os.path.join(self.img_dir, vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))
        # image_num = round(self.vid2durations[vid] - 1)    # equal to video total duration (seconds)

        with open(asr_file, "r") as f:
            subtitles = json.load(f)
            
        # parse timestamp
        cut_points = []         # G.T. cut point + a little bit offset
        #!! real_cut_points = []    # record G.T. cut point
        #!! descriptions = []
        for timestamp_str in timestamp:
            sec, description = extract_first_timestamp(timestamp_str)
                
            if sec < 4:
                continue
            # if sec > image_num - 4:
            if sec > image_num:
                continue
            
            cut_points.append(sec)
            #!! real_cut_points.append(sec)
            #!! descriptions.append(description)

        ### 1. segment timeline
        max_offset = 2      # positive clip if the distance to GT < 2 offset 
        clips = [[start_t, start_t + self.clip_frame_num] for start_t in range(0, image_num - self.clip_frame_num, 2 * max_offset)]
        assert clips[-1][1] <= image_num

        ### 2. calculate positive or negative
        clip_labels = []
        pos_clip_indices = []
        neg_clip_indices = []
        for idx, clip in enumerate(clips):
            start_t, end_t = clip
            label = 0
            for cut_point in cut_points:
                cut_point_start_t = cut_point - self.half_clip_frame_num
                cut_point_end_t = cut_point + self.half_clip_frame_num
                a = max(start_t, cut_point_start_t)
                mi = min(start_t, cut_point_start_t)
                b = min(end_t, cut_point_end_t)
                ma = max(end_t, cut_point_end_t)
                iou = (b - a) / (ma - mi) 
                if iou >= (self.clip_frame_num - max_offset) / (self.clip_frame_num + max_offset):
                    label = 1
                    break
            if label == 1:
                pos_clip_indices.append(idx)
            else:
                neg_clip_indices.append(idx)
            clip_labels.append(label)

        
        ### 3. sample positive or negative clip

        if not pos_clip_indices: # 모든 clip의 chapter 구분점이 없는 경우 negative로 sampling
            is_positive = 0
        else:
            is_positive = random.sample([0, 1], k=1)[0]

        if is_positive:
            pos_clip_index = random.sample(pos_clip_indices, k=1)[0]
            clip = clips[pos_clip_index]
        else:
            neg_clip_index = random.sample(neg_clip_indices, k=1)[0]
            clip = clips[neg_clip_index]

        # sampled clip, [start_second, end_second]
        clip_start_sec, clip_end_sec = clip

        # get the subtitle in-between [clip_start_sec - text_extra_time_gap, clip_end_sec + text_extra_time_gap]
        text_extra_time_gap = 1
        text_clip = "" 
        for sub in subtitles:
            text = sub["text"]
            start_sec = sub["start"]
            if clip_start_sec - text_extra_time_gap < start_sec < clip_end_sec + text_extra_time_gap:
                if len(text_clip) == 0:
                    text_clip += text
                else:
                    text_clip += " " + text
        
        # make labels
        label = 1 if is_positive else 0

        ### process text
        # put [CLS] at the first, so that we can train sentence representation after pretrained
        text_clip = "[CLS] " + text_clip # 해당 clip 내의 subtitle 추출
        tokens = self.tokenizer.tokenize(text_clip)
        # truncate
        tokens = tokens[:self.max_text_len]
        # pad
        attention_mask = [1] * len(tokens)
        if len(tokens) < self.max_text_len:
            zero_pad_list = [0] * (self.max_text_len - len(tokens))
            pad_list = ["[PAD]"] * (self.max_text_len - len(tokens))
            tokens += pad_list
            attention_mask += zero_pad_list

        # Convert token to vocabulary indices
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        text_ids = torch.from_numpy(np.array(ids)).long()
        attention_mask = torch.from_numpy(np.array(attention_mask)).long()
        
        
        if self.mode == "text":
            img_clip = 0    # dummy image
        else:
            # get images in this clip
            img_list = []
            for idx in range(clip_start_sec, clip_end_sec):
                # There is a bug ffmpeg extract frame, which causes the image misalign frame. 
                # We offset right frame 2 unit, but keep the beginning and the end not change.  
                if clip_start_sec <= 2 or clip_start_sec >= image_num - self.clip_frame_num - 2:
                    image_filename = "%05d.jpg"%(idx+1)
                else:
                    image_filename = "%05d.jpg"%(idx+3)
                image_filename = os.path.join(image_path, image_filename)
                img = Image.open(image_filename).convert('RGB')
                img = self.transform(img)
                img_list.append(img)

            img_clip = torch.stack(img_list, dim=0) # 해당 clip 내의 frame 추출

        return img_clip, text_ids, attention_mask, label

    def __len__(self):
        return len(self.vids)

class YoutubeAllClipDataset:
    def __init__(self, img_dir, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, mode="all", transform=None, target_transform=None):
        """
        Sample a positive or negative clip from dataset  
        Note that the video frame is sampled by 1 frame/second
        """
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.mode = mode        # text (text only), image (image only) or all (multiple-model)
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        self.img_dir = img_dir
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

        # asr files
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
        timestamp = self.vid2timestamps[vid]
        asr_file = self.vid2asr_files[vid]
        image_path = os.path.join(self.img_dir, vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))
        # image_num = round(self.vid2durations[vid] - 1)    # equal to video total duration (seconds)

        with open(asr_file, "r") as f:
            subtitles = json.load(f)
            
        # parse timestamp
        cut_points = []         # G.T. cut point + a little bit offset
        #!! real_cut_points = []    # record G.T. cut point
        #!! descriptions = []
        for timestamp_str in timestamp:
            sec, description = extract_first_timestamp(timestamp_str)
                
            if sec < 4:
                continue
            # if sec > image_num - 4:
            if sec > image_num:
                continue
            
            cut_points.append(sec)
            #!! real_cut_points.append(sec)
            #!! descriptions.append(description)

        ### 1. segment timeline
        max_offset = 2      # positive clip if the distance to GT < 2 offset 
        clips = [[start_t, start_t + self.clip_frame_num] for start_t in range(0, image_num - self.clip_frame_num, 16 * max_offset)]
        assert clips[-1][1] <= image_num

        ### 2. calculate positive or negative
        #!! clip_labels = []
        pos_clip_indices = []
        neg_clip_indices = []
        for idx, clip in enumerate(clips):
            start_t, end_t = clip
            label = 0
            for cut_point in cut_points:
                cut_point_start_t = cut_point - self.half_clip_frame_num
                cut_point_end_t = cut_point + self.half_clip_frame_num
                a = max(start_t, cut_point_start_t)
                mi = min(start_t, cut_point_start_t)
                b = min(end_t, cut_point_end_t)
                ma = max(end_t, cut_point_end_t)
                iou = (b - a) / (ma - mi) 
                if iou >= (self.clip_frame_num - max_offset) / (self.clip_frame_num + max_offset):
                    label = 1
                    break
            if label == 1:
                pos_clip_indices.append(idx)
            else:
                neg_clip_indices.append(idx)
            #!! clip_labels.append(label)

        
        ### 3. sample positive or negative clip
        #!!
        # Select target clip
        is_positive = random.choice([0, 1]) if pos_clip_indices else 0
        target_idx = random.choice(pos_clip_indices if is_positive else neg_clip_indices)

        all_clip_images = []
        all_text_ids = []
        all_attention_masks = []

        for clip_idx, clip in enumerate(clips):
            # Get images for current clip
            start_sec, end_sec = clip
            clip_imgs = []
            for idx in range(start_sec, end_sec):
                if start_sec <= 2 or start_sec >= image_num - self.clip_frame_num - 2:
                    image_filename = f"{idx+1:05d}.jpg"
                else:
                    image_filename = f"{idx+3:05d}.jpg"
                image_path = os.path.join(self.img_dir, vid, image_filename)
                img = Image.open(image_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                clip_imgs.append(img)
            all_clip_images.append(torch.stack(clip_imgs))

            # Get text for current clip
            text_clip = "[CLS] "
            text_extra_time_gap = 1
            for sub in subtitles:
                if start_sec - text_extra_time_gap < sub["start"] < end_sec + text_extra_time_gap:
                    text_clip += sub["text"] + " "

            # Tokenize text
            tokens = self.tokenizer.tokenize(text_clip)[:self.max_text_len]
            attention_mask = [1] * len(tokens)
            padding_length = self.max_text_len - len(tokens)
            if padding_length > 0:
                tokens.extend(["[PAD]"] * padding_length)
                attention_mask.extend([0] * padding_length)

            text_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            attention_mask = torch.tensor(attention_mask)
            
            all_text_ids.append(text_ids)
            all_attention_masks.append(attention_mask)

        # Stack all tensors
        img_clips = torch.stack(all_clip_images)  # [num_clips, num_frames, C, H, W]
        text_ids = torch.stack(all_text_ids)      # [num_clips, max_text_len]
        attention_masks = torch.stack(all_attention_masks)  # [num_clips, max_text_len]
        label = torch.tensor(1 if is_positive else 0)

        # print(f'img_clips: {img_clips.shape}')
        # print(f'text_ids: {text_ids.shape}')
        # print(f'attention_masks: {attention_masks.shape}')
        # print(f'label: {label}')
        # print(f'target_idx: {target_idx}')

        return img_clips, text_ids, attention_masks, label, torch.tensor(target_idx)

    def __len__(self):
        return len(self.vids)

class WindowClipDataset:
    def __init__(self, img_dir, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, window_size=2, mode="all", transform=None):
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.window_size = window_size
        self.mode = mode
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        self.img_dir = img_dir

        # Load video info
        all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)
        self.vid2title = {all_vids[i]: titles[i] for i in range(len(all_vids))}
        self.vid2timestamps = {all_vids[i]: timestamps[i] for i in range(len(all_vids))}
        self.vid2durations = {all_vids[i]: durations[i] for i in range(len(all_vids))}

        # Load video IDs
        with open(vid_file, "r") as f:
            self.vids = [x.strip() for x in f.readlines()]

        # Load ASR files
        asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
        self.vid2asr_files = {}
        for asr_file in asr_file_list:
            filename = os.path.basename(asr_file)
            vid = filename.split(".")[0][9:]
            self.vid2asr_files[vid] = asr_file

        self.transform = transform

    def __getitem__(self, i):
        vid = self.vids[i]
        timestamp = self.vid2timestamps[vid]
        asr_file = self.vid2asr_files[vid]
        image_path = os.path.join(self.img_dir, vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))

        with open(asr_file, "r") as f:
            subtitles = json.load(f)
            
        # Get cut points
        cut_points = []
        for timestamp_str in timestamp:
            sec, _ = extract_first_timestamp(timestamp_str)
            if sec < 4 or sec > image_num:
                continue
            cut_points.append(sec)

        # Segment into clips
        max_offset = 2    
        clips = [[start_t, start_t + self.clip_frame_num] 
                for start_t in range(0, image_num - self.clip_frame_num, 2 * max_offset)]
        
        # Get positive/negative clips
        pos_clip_indices = []
        neg_clip_indices = []
        for idx, clip in enumerate(clips):
            start_t, end_t = clip
            label = 0
            for cut_point in cut_points:
                cut_point_start_t = cut_point - self.half_clip_frame_num
                cut_point_end_t = cut_point + self.half_clip_frame_num
                a = max(start_t, cut_point_start_t)
                mi = min(start_t, cut_point_start_t)
                b = min(end_t, cut_point_end_t)
                ma = max(end_t, cut_point_end_t)
                iou = (b - a) / (ma - mi) 
                if iou >= (self.clip_frame_num - max_offset) / (self.clip_frame_num + max_offset):
                    label = 1
                    break
            if label == 1:
                pos_clip_indices.append(idx)
            else:
                neg_clip_indices.append(idx)

        # Select target clip
        is_positive = random.choice([0, 1]) if pos_clip_indices else 0
        target_idx = random.choice(pos_clip_indices if is_positive else neg_clip_indices)

        # Get window clips
        window_indices = []
        skip_size = self.clip_frame_num // (2 * max_offset)  
        for idx in range(target_idx - skip_size * self.window_size, target_idx + skip_size * self.window_size + 1, skip_size):
            if 0 <= idx < len(clips):
                window_indices.append(idx)
            else:
                window_indices.append(-1)  # Padding index

        # print(f'target idx: {target_idx}, window idx: {window_indices}')

        # Process clips within window
        all_clip_images = []
        all_text_ids = []
        all_attention_masks = []

        for idx in window_indices:
            if idx == -1:
                # Add zero padding
                if self.mode != "text":
                    padding_img = torch.zeros((self.clip_frame_num, 3, 224, 224))
                    all_clip_images.append(padding_img)
                
                padding_text = torch.zeros(self.max_text_len, dtype=torch.long)
                padding_mask = torch.zeros(self.max_text_len, dtype=torch.long)
                
                all_text_ids.append(padding_text)
                all_attention_masks.append(padding_mask)
                continue

            clip = clips[idx]
            start_sec, end_sec = clip

            # Process images
            if self.mode != "text":
                clip_imgs = []
                for frame_idx in range(start_sec, end_sec):
                    if start_sec <= 2 or start_sec >= image_num - self.clip_frame_num - 2:
                        image_filename = f"{frame_idx+1:05d}.jpg"
                    else:
                        image_filename = f"{frame_idx+3:05d}.jpg"
                    image_path = os.path.join(self.img_dir, vid, image_filename)
                    with Image.open(image_path) as img:  # context manager 사용
                        img = img.convert('RGB')
                    # img = Image.open(image_path).convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        clip_imgs.append(img)
                all_clip_images.append(torch.stack(clip_imgs))

            # Process text
            text_clip = "[CLS] "
            for sub in subtitles:
                if start_sec - 1 < sub["start"] < end_sec + 1:
                    text_clip += sub["text"] + " "

            tokens = self.tokenizer.tokenize(text_clip)[:self.max_text_len]
            attention_mask = [1] * len(tokens)
            padding_length = self.max_text_len - len(tokens)
            if padding_length > 0:
                tokens.extend(["[PAD]"] * padding_length)
                attention_mask.extend([0] * padding_length)

            text_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            attention_mask = torch.tensor(attention_mask)
            
            all_text_ids.append(text_ids)
            all_attention_masks.append(attention_mask)

        # Stack all tensors
        if self.mode == "text":
            img_clips = torch.tensor(0)
        else:
            img_clips = torch.stack(all_clip_images)  # [window_size*2+1, num_frames, C, H, W]
            
        text_ids = torch.stack(all_text_ids)          # [window_size*2+1, max_text_len]
        attention_masks = torch.stack(all_attention_masks)  # [window_size*2+1, max_text_len]
        label = torch.tensor(1 if is_positive else 0)
        center_idx = self.window_size  # target clip은 항상 중앙에 위치)

        clip_start_frames = []
        for idx in window_indices:
            if idx == -1:  # padding index
                clip_start_frames.append(-1)  # padding case
            else:
                clip_start_frames.append(clips[idx][0])  # 각 클립의 시작 프레임

        clips_info = {
            'clip_start_frame': torch.tensor(clip_start_frames),  # 윈도우 내 각 클립의 시작 프레임
            'total_frames': torch.tensor(image_num),              # 비디오의 총 프레임 수
            'target_clip_idx': torch.tensor(target_idx),          # 타겟 클립의 인덱스
            'total_num_clips': torch.tensor(len(clips))         # 비디오의 총 클립 수
        }

        return img_clips, text_ids, attention_masks, label, clips_info

    def __len__(self):
        return len(self.vids)


class WindowClipDatasetv2:
    def __init__(self, img_dir, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, window_size=2, mode="all", transform=None):
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.window_size = window_size
        self.mode = mode
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        self.img_dir = img_dir
        self.transform = transform
        self.max_offset = 2

        # Load video info
        all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)
        self.vid2title = {all_vids[i]: titles[i] for i in range(len(all_vids))}
        self.vid2timestamps = {all_vids[i]: timestamps[i] for i in range(len(all_vids))}
        self.vid2durations = {all_vids[i]: durations[i] for i in range(len(all_vids))}

        # Load video IDs
        with open(vid_file, "r") as f:
            self.vids = [x.strip() for x in f.readlines()]

        # Load ASR files
        asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
        self.vid2asr_files = {}
        for asr_file in asr_file_list:
            filename = os.path.basename(asr_file)
            vid = filename.split(".")[0][9:]
            self.vid2asr_files[vid] = asr_file

        # Pre-process video information
        self.vid2clips = {}  # 비디오별 클립 정보 저장
        self.vid2cutpoints = {}  # 비디오별 cut point 저장
        self.vid2numframes = {}  # 비디오별 총 프레임 수 저장
        self._preprocess_videos()

        # Initialize image memory mapping
        self.image_memmap = {}
        if self.mode != "text":
            self._setup_image_memmap()

    def _preprocess_videos(self):
        """비디오별 클립 정보 사전 처리"""
        for vid in self.vids:
            # Get timestamps and ASR
            timestamp = self.vid2timestamps[vid]
            with open(self.vid2asr_files[vid], "r") as f:
                subtitles = json.load(f)

            # Get cut points
            image_path = os.path.join(self.img_dir, vid)
            image_num = len(glob.glob(image_path + "/*.jpg"))
            self.vid2numframes[vid] = image_num

            cut_points = []
            for timestamp_str in timestamp:
                sec, _ = extract_first_timestamp(timestamp_str)
                if 4 <= sec <= image_num - 4:
                    cut_points.append(sec)
            self.vid2cutpoints[vid] = cut_points

            # Generate clips
            clips = []
            for start_t in range(0, image_num - self.clip_frame_num, 2 * self.max_offset):
                end_t = start_t + self.clip_frame_num
                clip_info = {
                    "start": start_t,
                    "end": end_t,
                    "text": self._get_clip_text(subtitles, start_t, end_t),
                    "label": self._check_clip_label(start_t, end_t, cut_points)
                }
                clips.append(clip_info)
            self.vid2clips[vid] = clips

    def _get_clip_text(self, subtitles, start_sec, end_sec, extra_time_gap=1):
        """클립에 해당하는 자막 추출"""
        text_clip = []
        for sub in subtitles:
            if start_sec - extra_time_gap < sub["start"] < end_sec + extra_time_gap:
                text_clip.append(sub["text"])
        return " ".join(text_clip)

    def _check_clip_label(self, start_t, end_t, cut_points):
        """클립이 positive인지 확인"""
        for cp in cut_points:
            cut_point_start_t = cp - self.half_clip_frame_num
            cut_point_end_t = cp + self.half_clip_frame_num
            a = max(start_t, cut_point_start_t)
            mi = min(start_t, cut_point_start_t)
            b = min(end_t, cut_point_end_t)
            ma = max(end_t, cut_point_end_t)
            iou = (b - a) / (ma - mi)
            if iou >= (self.clip_frame_num - self.max_offset) / (self.clip_frame_num + self.max_offset):
                return 1
        return 0

    def _setup_image_memmap(self):
        """비디오 프레임 메모리 매핑 설정"""
        for vid in self.vids:
            memmap_path = os.path.join(self.img_dir, f"{vid}_frames.mmap")
            if not os.path.exists(memmap_path):
                # Create memory mapped array
                image_num = self.vid2numframes[vid]
                shape = (image_num, 3, 224, 224)  # Assuming standard image size
                mmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=shape)
                
                # Load images into memmap
                for frame_idx in range(image_num):
                    image_path = os.path.join(self.img_dir, vid, f"{frame_idx+1:05d}.jpg")
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        mmap[frame_idx] = img.numpy()
                mmap.flush()
            
            # Open memmap in read mode
            self.image_memmap[vid] = np.memmap(
                memmap_path,
                dtype='float32',
                mode='r',
                shape=(self.vid2numframes[vid], 3, 224, 224)
            )
    
    def __getitem__(self, i):
        vid = self.vids[i]
        clips = self.vid2clips[vid]

        # Select target clip and get window indices
        is_positive = random.choice([0, 1])
        pos_indices = [i for i, c in enumerate(clips) if c["label"] == 1]
        neg_indices = [i for i, c in enumerate(clips) if c["label"] == 0]
        
        if not pos_indices:  # if no positive clips
            is_positive = 0

        target_idx = random.choice(pos_indices if is_positive else neg_indices)
        
        # Get window clips
        window_indices = []
        skip_size = self.clip_frame_num // (2 * self.max_offset)
        for idx in range(target_idx - skip_size * self.window_size,
                        target_idx + skip_size * self.window_size + 1,
                        skip_size):
            if 0 <= idx < len(clips):
                window_indices.append(idx)
            else:
                window_indices.append(-1)

        # Process clips within window
        all_clip_images = []
        all_text_ids = []
        all_attention_masks = []
        
        for idx in window_indices:
            if idx == -1:
                # Add zero padding
                if self.mode != "text":
                    padding_img = torch.zeros((self.clip_frame_num, 3, 224, 224))
                    all_clip_images.append(padding_img)
                
                padding_text = torch.zeros(self.max_text_len, dtype=torch.long)
                padding_mask = torch.zeros(self.max_text_len, dtype=torch.long)
                
                all_text_ids.append(padding_text)
                all_attention_masks.append(padding_mask)
                continue

            clip = clips[idx]
            start_sec, end_sec = clip["start"], clip["end"]

            # Process images
            if self.mode != "text":
                if vid in self.image_memmap:
                    # Load from memmap
                    frames = torch.from_numpy(
                        self.image_memmap[vid][start_sec:end_sec]
                    ).float()
                    all_clip_images.append(frames)
                else:
                    # Load from disk
                    clip_imgs = []
                    for frame_idx in range(start_sec, end_sec):
                        image_path = os.path.join(
                            self.img_dir,
                            vid,
                            f"{frame_idx+1:05d}.jpg"
                        )
                        with Image.open(image_path) as img:
                            img = img.convert('RGB')
                            if self.transform:
                                img = self.transform(img)
                            clip_imgs.append(img)
                    all_clip_images.append(torch.stack(clip_imgs))

            # Process text
            text_clip = "[CLS] " + clip["text"]
            tokens = self.tokenizer.tokenize(text_clip)[:self.max_text_len]
            attention_mask = [1] * len(tokens)
            
            padding_length = self.max_text_len - len(tokens)
            if padding_length > 0:
                tokens.extend(["[PAD]"] * padding_length)
                attention_mask.extend([0] * padding_length)

            text_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            attention_mask = torch.tensor(attention_mask)
            
            all_text_ids.append(text_ids)
            all_attention_masks.append(attention_mask)

        # Stack tensors
        if self.mode == "text":
            img_clips = torch.tensor(0)
        else:
            img_clips = torch.stack(all_clip_images)
            
        text_ids = torch.stack(all_text_ids)
        attention_masks = torch.stack(all_attention_masks)
        label = torch.tensor(1 if is_positive else 0)

        # Prepare clip info
        clip_start_frames = []
        for idx in window_indices:
            if idx == -1:
                clip_start_frames.append(-1)
            else:
                clip_start_frames.append(clips[idx]["start"])

        clips_info = {
            'clip_start_frame': torch.tensor(clip_start_frames),
            'total_frames': torch.tensor(self.vid2numframes[vid]),
            'target_clip_idx': torch.tensor(target_idx),
            'total_num_clips': torch.tensor(len(clips)),
            'video_id': vid
        }

        return img_clips, text_ids, attention_masks, label, clips_info

    def __len__(self):
        return len(self.vids)

    def cleanup(self):
        """리소스 정리"""
        for mmap in self.image_memmap.values():
            del mmap
        self.image_memmap.clear()

'''class WindowClipDataset:
    def __init__(self, img_dir, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, window_size=2, mode="all", transform=None):
        print("Initializing WindowClipDataset...")
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.window_size = window_size
        self.mode = mode
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        self.img_dir = img_dir
        self.transform = transform
        self.max_offset = 2

        # Load video info
        print("Loading video info...")
        all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)
        self.vid2title = {all_vids[i]: titles[i] for i in range(len(all_vids))}
        self.vid2timestamps = {all_vids[i]: timestamps[i] for i in range(len(all_vids))}
        self.vid2durations = {all_vids[i]: durations[i] for i in range(len(all_vids))}

        # Load video IDs
        print("Loading video IDs...")
        with open(vid_file, "r") as f:
            self.vids = [x.strip() for x in f.readlines()]

        # Pre-process and cache video information
        print("Pre-processing video information...")
        self.vid2clips = {}
        self.vid2cutpoints = {}
        self.vid2numframes = {}
        
        for vid in self.vids:
            image_path = os.path.join(self.img_dir, vid)
            frame_files = glob.glob(os.path.join(image_path, "*.jpg"))
            if not frame_files:
                print(f"Warning: No frames found for video {vid}")
                continue
                
            image_num = len(frame_files)
            self.vid2numframes[vid] = image_num
            
            # Get timestamps
            timestamp = self.vid2timestamps[vid]
            cut_points = []
            for timestamp_str in timestamp:
                sec, _ = extract_first_timestamp(timestamp_str)
                if 4 <= sec <= image_num - 4:
                    cut_points.append(sec)
            self.vid2cutpoints[vid] = cut_points
            
            # Generate clips
            clips = []
            for start_t in range(0, image_num - self.clip_frame_num, 2 * self.max_offset):
                end_t = start_t + self.clip_frame_num
                if end_t > image_num:
                    break
                clips.append((start_t, end_t))
            self.vid2clips[vid] = clips

        print("Initialization complete")
    '''
def custom_collate_fn(batch):
    # Get max lengths for padding
    max_clips = max([b[0].size(0) for b in batch])
    
    # Initialize lists for batched data
    img_clips, text_ids, attention_masks = [], [], []
    labels, clip_infos = [], []
    
    for img_c, txt_id, att_mask, label, clip_info in batch:
        # Get current sizes
        num_clips = img_c.size(0)
        
        # Handle image clips padding
        if num_clips < max_clips:
            padding = torch.zeros((max_clips - num_clips, *img_c.size()[1:]), dtype=img_c.dtype)
            img_c = torch.cat([img_c, padding], dim=0)
            
            # Handle text padding
            text_padding = torch.zeros((max_clips - num_clips, *txt_id.size()[1:]), dtype=txt_id.dtype)
            txt_id = torch.cat([txt_id, text_padding], dim=0)
            att_mask = torch.cat([att_mask, text_padding], dim=0)
            
            # Update clip info with padding
            if 'clip_start_frame' in clip_info:
                clip_info['clip_start_frame'] = F.pad(
                    clip_info['clip_start_frame'], 
                    (0, max_clips - num_clips), 
                    value=-1
                )
            
        img_clips.append(img_c)
        text_ids.append(txt_id)
        attention_masks.append(att_mask)
        labels.append(label)
        clip_infos.append(clip_info)
    
    # Stack all tensors
    batch_clips_info = {}
    for k in clip_infos[0].keys():
        if k == 'video_id':  # Handle string field separately
            batch_clips_info[k] = [info[k] for info in clip_infos]
        else:  # Handle tensor fields
            batch_clips_info[k] = torch.stack([info[k] for info in clip_infos])
    
    return (
        torch.stack(img_clips),
        torch.stack(text_ids), 
        torch.stack(attention_masks),
        torch.stack(labels),
        batch_clips_info
    )
class WindowClipIDDataset:
    def __init__(self, img_dir, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, window_size=2, mode="all", transform=None):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.half_clip_frame_num = int(clip_frame_num // 2)  # Added this line
        self.max_text_len = max_text_len
        self.window_size = window_size
        self.mode = mode
        self.transform = transform
        self.max_offset = 2

        # Load video info
        all_vids, _, _, timestamps = parse_csv_to_list(data_file)
        self.vid2timestamps = {all_vids[i]: timestamps[i] for i in range(len(all_vids))}
        
        # Load video IDs and ASR files
        with open(vid_file, "r") as f:
            self.vids = [x.strip() for x in f]
            
        # Load ASR files
        asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
        self.vid2asr_files = {}
        for asr_file in asr_file_list:
            filename = os.path.basename(asr_file)
            vid = filename.split(".")[0][9:]
            self.vid2asr_files[vid] = asr_file
            
        # Process clips
        self.vid2clips = {}
        self.vid2numframes = {}
        for vid in self.vids:
            image_path = os.path.join(img_dir, vid)
            image_num = len(glob.glob(os.path.join(image_path, "*.jpg")))
            self.vid2numframes[vid] = image_num
            
            # Generate clips
            self.vid2clips[vid] = [
                [start_t, start_t + clip_frame_num] 
                for start_t in range(0, image_num - clip_frame_num, 2 * self.max_offset)
            ]

    def load_frame(self, vid, frame_idx):
        image_path = os.path.join(self.img_dir, vid, f"{frame_idx+1:05d}.jpg")
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img

    def get_clip_info(self, i):
        vid = self.vids[i]
        clips = self.vid2clips[vid]
        cut_points = self.vid2timestamps[vid]
        image_num = self.vid2numframes[vid]

        # Select target clip
        pos_indices = []
        neg_indices = []
        
        for idx, (start_t, end_t) in enumerate(clips):
            label = 0
            for timestamp_str in cut_points:
                sec, _ = extract_first_timestamp(timestamp_str)
                if sec < 4 or sec > image_num:
                    continue
                    
                pos_st = sec - self.half_clip_frame_num
                pos_et = sec + self.half_clip_frame_num
                a = max(start_t, pos_st)
                mi = min(start_t, pos_st)
                b = min(end_t, pos_et)
                ma = max(end_t, pos_et)
                iou = (b - a) / (ma - mi)
                
                if iou >= (self.clip_frame_num - self.max_offset) / (self.clip_frame_num + self.max_offset):
                    label = 1
                    break
                    
            if label == 1:
                pos_indices.append(idx)
            else:
                neg_indices.append(idx)

        is_positive = random.choice([0, 1]) if pos_indices else 0
        target_idx = random.choice(pos_indices if is_positive else neg_indices)
        
        clip_info = {
            "vid": vid,
            "clip_start_end": clips[target_idx],
            "label": 1 if is_positive else 0
        }

        # Calculate window indices
        window_indices = []
        skip_size = self.clip_frame_num // (2 * self.max_offset)
        start_idx = target_idx - skip_size * self.window_size
        end_idx = target_idx + skip_size * self.window_size + 1
        
        for idx in range(start_idx, end_idx, skip_size):
            window_indices.append(idx if 0 <= idx < len(clips) else -1)

        return clip_info, window_indices

    def __getitem__(self, i):
        vid = self.vids[i]
        clips = self.vid2clips[vid]
        asr_file = self.vid2asr_files[vid]
        
        with open(asr_file, "r") as f:
            subtitles = json.load(f)
            
        clip_info, window_indices = self.get_clip_info(i)
        start_sec, end_sec = clip_info["clip_start_end"]

        all_clip_images = []
        all_text_ids = []
        all_attention_masks = []

        for idx in window_indices:
            if idx == -1:
                # Add padding
                if self.mode != "text":
                    all_clip_images.append(torch.zeros((self.clip_frame_num, 3, 224, 224)))
                all_text_ids.append(torch.zeros(self.max_text_len, dtype=torch.long))
                all_attention_masks.append(torch.zeros(self.max_text_len, dtype=torch.long))
                continue

            curr_clip = clips[idx]
            curr_start, curr_end = curr_clip

            # Process text
            text_clip = "[CLS] "
            for sub in subtitles:
                if curr_start - 1 < sub["start"] < curr_end + 1:
                    text_clip += sub["text"] + " "

            tokens = self.tokenizer.tokenize(text_clip)[:self.max_text_len]
            attention_mask = [1] * len(tokens)
            padding_length = self.max_text_len - len(tokens)
            if padding_length > 0:
                tokens.extend(["[PAD]"] * padding_length)
                attention_mask.extend([0] * padding_length)

            text_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            all_text_ids.append(text_ids)
            all_attention_masks.append(torch.tensor(attention_mask))

            # Process images
            if self.mode != "text":
                frames = []
                for frame_idx in range(curr_start, curr_end):
                    frame = self.load_frame(vid, frame_idx)
                    frames.append(frame)
                all_clip_images.append(torch.stack(frames))

        # Stack tensors
        if self.mode == "text":
            img_clips = torch.tensor(0)
        else:
            img_clips = torch.stack(all_clip_images)

        text_ids = torch.stack(all_text_ids)
        attention_masks = torch.stack(all_attention_masks)

        clips_info = {
            'clip_start_frame': torch.tensor([clips[idx][0] if idx != -1 else -1 for idx in window_indices]),
            'total_frames': torch.tensor(self.vid2numframes[vid]),
            'target_clip_idx': torch.tensor(window_indices[self.window_size]),
            'total_num_clips': torch.tensor(len(clips)),
            'video_id': vid
        }

        return img_clips, text_ids, attention_masks, torch.tensor(clip_info["label"]), clips_info
    '''def __getitem__(self, i):
        vid = self.vids[i]
        clips = self.vid2clips[vid]
        cut_points = self.vid2cutpoints[vid]
        image_num = self.vid2numframes[vid]
        
        # Select target clip
        pos_indices = []
        neg_indices = []
        
        for idx, (start_t, end_t) in enumerate(clips):
            label = 0
            for cp in cut_points:
                pos_st = cp - self.half_clip_frame_num
                pos_et = cp + self.half_clip_frame_num
                a = max(start_t, pos_st)
                mi = min(start_t, pos_st)
                b = min(end_t, pos_et)
                ma = max(end_t, pos_et)
                iou = (b - a) / (ma - mi)
                
                if iou >= (self.clip_frame_num - self.max_offset) / (self.clip_frame_num + self.max_offset):
                    label = 1
                    break
                    
            if label == 1:
                pos_indices.append(idx)
            else:
                neg_indices.append(idx)
                
        is_positive = random.choice([0, 1]) if pos_indices else 0
        target_idx = random.choice(pos_indices if is_positive else neg_indices)
        
        # Process window clips
        all_clip_images = []
        all_text_ids = []
        all_attention_masks = []
        clip_start_frames = []
        
        window_indices = []
        skip_size = self.clip_frame_num // (2 * self.max_offset)
        for idx in range(target_idx - skip_size * self.window_size,
                        target_idx + skip_size * self.window_size + 1,
                        skip_size):
            window_indices.append(idx if 0 <= idx < len(clips) else -1)
            
        for idx in window_indices:
            if idx == -1:
                # Add padding
                if self.mode != "text":
                    all_clip_images.append(torch.zeros((self.clip_frame_num, 3, 224, 224)))
                all_text_ids.append(torch.zeros(self.max_text_len, dtype=torch.long))
                all_attention_masks.append(torch.zeros(self.max_text_len, dtype=torch.long))
                clip_start_frames.append(-1)
                continue
                
            start_sec, end_sec = clips[idx]
            clip_start_frames.append(start_sec)

            # Process images
            if self.mode != "text":
                clip_imgs = []
                for frame_idx in range(start_sec, end_sec):
                    frame_path = os.path.join(self.img_dir, vid, f"{frame_idx+1:05d}.jpg")
                    try:
                        with Image.open(frame_path) as img:
                            img = img.convert('RGB')
                            if self.transform:
                                img = self.transform(img)
                            clip_imgs.append(img)
                    except Exception as e:
                        print(f"Error loading frame {frame_path}: {e}")
                        raise
                all_clip_images.append(torch.stack(clip_imgs))

            # Process text
            subtitle = "" # Get subtitle for this clip window
            text_clip = "[CLS] " + subtitle
            tokens = self.tokenizer.tokenize(text_clip)[:self.max_text_len]
            attention_mask = [1] * len(tokens)
            
            if len(tokens) < self.max_text_len:
                tokens.extend(["[PAD]"] * (self.max_text_len - len(tokens)))
                attention_mask.extend([0] * (self.max_text_len - len(tokens)))
                
            text_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            all_text_ids.append(text_ids)
            all_attention_masks.append(torch.tensor(attention_mask))

        # Prepare final tensors
        if self.mode == "text":
            img_clips = torch.tensor(0)
        else:
            img_clips = torch.stack(all_clip_images)
            
        text_ids = torch.stack(all_text_ids)
        attention_masks = torch.stack(all_attention_masks)
        label = torch.tensor(1 if is_positive else 0)

        clips_info = {
            'clip_start_frame': torch.tensor(clip_start_frames),
            'total_frames': torch.tensor(image_num),
            'target_clip_idx': torch.tensor(target_idx),
            'total_num_clips': torch.tensor(len(clips)),
            'video_id': vid
        }

        return img_clips, text_ids, attention_masks, label, clips_info'''

    def __len__(self):
        return len(self.vids)

    def cleanup(self):
        print("Cleaning up resources...")
        # Clear any cached data
        self.vid2clips.clear()
        self.vid2cutpoints.clear()
        self.vid2numframes.clear()

class YoutubeListwiseClipDataset:
    def __init__(self, img_dir, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, negative_clip_num=10, mode="all", transform=None, target_transform=None):
        """
        Sample 1 positive and multiple negative clips within a video from dataset
        Note that the video frame is sampled by 1 frame/second
        """
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.negative_clip_num = negative_clip_num
        self.mode = mode        # text (text only), image (image only) or all (multiple-model)
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        self.img_dir = img_dir
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

        # asr files
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
        timestamp = self.vid2timestamps[vid]
        asr_file = self.vid2asr_files[vid]
        image_path = os.path.join(self.img_dir, vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))
        # image_num = round(self.vid2durations[vid] - 1)    # equal to video total duration (seconds)

        with open(asr_file, "r") as f:
            subtitles = json.load(f)

        # parse timestamp
        cut_points = []         # G.T. cut point + a little bit offset
        real_cut_points = []    # record G.T. cut point
        descriptions = []
        for timestamp_str in timestamp:
            sec, description = extract_first_timestamp(timestamp_str)
            if sec < 4:         # discard the chapter start time is close to beginning and end
                continue
            if sec > image_num - 4:
                continue
            
            cut_points.append(sec)
            real_cut_points.append(sec)
            descriptions.append(description)

        ### 1. segment timeline
        max_offset = 2
        clips = [[start_t, start_t + self.clip_frame_num] for start_t in range(0, image_num - self.clip_frame_num, 2 * max_offset)]
        assert clips[-1][1] <= image_num

        ### 2. calculate positive or negative
        clip_labels = []
        pos_clip_indices = []
        neg_clip_indices = []
        for idx, clip in enumerate(clips):
            start_t, end_t = clip
            label = 0
            for cut_point in cut_points:
                cut_point_start_t = cut_point - self.half_clip_frame_num
                cut_point_end_t = cut_point + self.half_clip_frame_num
                a = max(start_t, cut_point_start_t)
                mi = min(start_t, cut_point_start_t)
                b = min(end_t, cut_point_end_t)
                ma = max(end_t, cut_point_end_t)
                iou = (b - a) / (ma - mi) 
                if iou >= (self.clip_frame_num - max_offset) / (self.clip_frame_num + max_offset):
                    label = 1
                    break
            if label == 1:
                pos_clip_indices.append(idx)
            else:
                neg_clip_indices.append(idx)
            clip_labels.append(label)

        ### 3. sample positive and negative clips
        pos_clip_index = random.sample(pos_clip_indices, k=2)
        pos_clips = [clips[idx] for idx in pos_clip_index] 
        neg_clip_index = random.sample(neg_clip_indices, k=self.negative_clip_num)
        neg_clips = [clips[idx] for idx in neg_clip_index] 

        # get data
        sampled_clips = pos_clips + neg_clips
        img_clips = []
        clip_text_ids = []
        attention_masks = []
        labels = [1 for idx in range(len(pos_clip_index))] + [0 for idx in range(len(neg_clip_index))]
        for clip_idx, clip in enumerate(sampled_clips):
            clip_start_sec, clip_end_sec = clip

            # get the subtitle in-between [clip_start_sec - text_extra_time_gap, clip_end_sec + text_extra_time_gap]
            text_extra_time_gap = 1
            text_clip = ""
            for sub_idx, sub in enumerate(subtitles):
                text = sub["text"]
                start_sec = sub["start"]
                
                if clip_start_sec - text_extra_time_gap < start_sec < clip_end_sec + text_extra_time_gap:
                    if len(text_clip) == 0:
                        text_clip += text
                    else:
                        text_clip += " " + text

            ### process text
            # put [CLS] at the first, so that we can train sentence representation after pretrained
            text_clip = "[CLS] " + text_clip
            tokens = self.tokenizer.tokenize(text_clip)
            # truncate
            tokens = tokens[:self.max_text_len]
            # pad
            attention_mask = [1] * len(tokens)
            if len(tokens) < self.max_text_len:
                zero_pad_list = [0] * (self.max_text_len - len(tokens))
                pad_list = ["[PAD]"] * (self.max_text_len - len(tokens))
                tokens += pad_list
                attention_mask += zero_pad_list

            # Convert token to vocabulary indices
            ids = self.tokenizer.convert_tokens_to_ids(tokens)

            clip_text_id = torch.from_numpy(np.array(ids)).long()
            attention_mask = torch.from_numpy(np.array(attention_mask)).long()
            
            if self.mode == "text":
                img_clip = 0    # dummy image
                img_clip = np.array(img_clip)
            else:
                # get images in this clip
                img_list = []
                for idx in range(clip_start_sec, clip_end_sec):
                    # There is a bug ffmpeg extract frame, which causes the image misalign frame. 
                    # We offset right frame 2 unit, but keep the beginning and the end not change.  
                    if clip_start_sec <= 2 or clip_start_sec >= image_num - self.clip_frame_num - 2:
                        image_filename = "%05d.jpg"%(idx+1)
                    else:
                        image_filename = "%05d.jpg"%(idx+3)
                    image_filename = os.path.join(image_path, image_filename)
                    img = Image.open(image_filename).convert('RGB')
                    img = np.array(img)
                    img_list.append(img)
                img_clip = np.stack(img_list, axis=0)

                # visualize this clip
                # print(f"https://www.youtube.com/watch?v={vid}&t={t}")
                # print(f"{clip_start_sec} - {clip_end_sec}")
                # print(text_clip)
                # print(label)

                # h, w, c = img.shape
                # clip_whole_image = np.zeros((h, w*len(img_list), c), dtype=np.uint8)
                # for idx, im in enumerate(img_list):
                #     clip_whole_image[:, idx*w:(idx+1)*w, :] = im
                # im = Image.fromarray(clip_whole_image)
                # im.save("./clip.jpg")
                # # plt.imshow(clip_whole_image)
                # # plt.show()
            
            # visualize text
            # print(f"vid {vid}, label {labels[clip_idx]}, {clip_start_sec} - {clip_end_sec}, {text_clip}")

            img_clip = torch.from_numpy(img_clip).float()
            img_clips.append(img_clip)
            clip_text_ids.append(clip_text_id)
            attention_masks.append(attention_mask)

        # print("============================")
        img_clips = torch.stack(img_clips, dim=0)
        clip_text_ids = torch.stack(clip_text_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        labels = torch.from_numpy(np.array(labels)).long()

        return img_clips, clip_text_ids, attention_masks, labels

    def __len__(self):
        return len(self.vids)

def custom_collate_fn(batch):
    # Get max lengths for padding
    max_clips = max([b[0].size(0) for b in batch])
    
    # Initialize lists for batched data
    img_clips, text_ids, attention_masks = [], [], []
    labels, clip_infos = [], []
    
    for img_c, txt_id, att_mask, label, clip_info in batch:
        # Get current sizes
        num_clips = img_c.size(0)
        
        # Handle image clips padding
        if num_clips < max_clips:
            padding = torch.zeros((max_clips - num_clips, *img_c.size()[1:]), dtype=img_c.dtype)
            img_c = torch.cat([img_c, padding], dim=0)
            
            # Handle text padding
            text_padding = torch.zeros((max_clips - num_clips, *txt_id.size()[1:]), dtype=txt_id.dtype)
            txt_id = torch.cat([txt_id, text_padding], dim=0)
            att_mask = torch.cat([att_mask, text_padding], dim=0)
            
            # Update clip info with padding
            if 'clip_start_frame' in clip_info:
                clip_info['clip_start_frame'] = F.pad(
                    clip_info['clip_start_frame'], 
                    (0, max_clips - num_clips), 
                    value=-1
                )
            
        img_clips.append(img_c)
        text_ids.append(txt_id)
        attention_masks.append(att_mask)
        labels.append(label)
        clip_infos.append(clip_info)
    
    # Stack all tensors
    batch_clips_info = {}
    for k in clip_infos[0].keys():
        if k == 'video_id':  # Handle string field separately
            batch_clips_info[k] = [info[k] for info in clip_infos]
        else:  # Handle tensor fields
            batch_clips_info[k] = torch.stack([info[k] for info in clip_infos])
    
    return (
        torch.stack(img_clips),
        torch.stack(text_ids), 
        torch.stack(attention_masks),
        torch.stack(labels),
        batch_clips_info
    )
'''def custom_collate_fn(batch):
    max_clips = max([b[0].size(0) for b in batch])  # 배치 내 최대 clip 수
    
    # Pad tensors to max length
    img_clips = []
    text_ids = []
    attention_masks = []
    labels = []
    target_idx = []
    
    for img_c, txt_id, att_mask, lab, tgt_idx in batch:
        num_clips = img_c.size(0)
        # Padding for images
        if num_clips < max_clips:
            padding = torch.zeros((max_clips - num_clips, *img_c.size()[1:]), dtype=img_c.dtype)
            img_c = torch.cat([img_c, padding], dim=0)
        img_clips.append(img_c)
        
        # Padding for text
        if num_clips < max_clips:
            padding = torch.zeros((max_clips - num_clips, *txt_id.size()[1:]), dtype=txt_id.dtype)
            txt_id = torch.cat([txt_id, padding], dim=0)
            padding = torch.zeros((max_clips - num_clips, *att_mask.size()[1:]), dtype=att_mask.dtype)
            att_mask = torch.cat([att_mask, padding], dim=0)
        text_ids.append(txt_id)
        attention_masks.append(att_mask)
        
        labels.append(lab)
        target_idx.append(tgt_idx)
    
    return (torch.stack(img_clips), 
            torch.stack(text_ids),
            torch.stack(attention_masks),
            torch.stack(labels),
            torch.stack(target_idx))'''


if __name__ == "__main__":
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    # from common_utils import set_random_seed
    # set_random_seed.use_fix_random_seed()
    # img_dir = "D:/youtube_video_frame_minidataset"
    # data_file = "D:/py3_code/video_chapter_youtube_dataset/dataset/test_mini_dataset.csv"
    img_dir = "/opt/tiger/youtube_video_frame_dataset"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    glove_pickle_file = "/opt/tiger/video_chapter_generation/glove.840B.300d.pickle"
    

    train_vision_preprocess = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter()], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = YoutubeClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num=8, max_text_len=100, mode="all", transform=train_vision_preprocess)
    # dataset = YoutubeIntraClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num=8, max_text_len=50, mode="text")
    # dataset = YoutubeListwiseClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num=8, max_text_len=50, negative_clip_num=10, mode="text")
    data_loader_params = dict(batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    data_loader = DataLoader(dataset, **data_loader_params)

    import time
    for img_clip, text_ids, attention_mask, label in data_loader:
        st = time.time()
        print(img_clip.size())
        print(label.size())
        et = time.time()
        print(f"cost time {et - st} ")