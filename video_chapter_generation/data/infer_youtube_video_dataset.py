"""
Dataset for infering youtube video.
It is used only on inference stage for mimic video chapter generation

"""


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
from transformers import OpenAIGPTTokenizer, BertTokenizer
from typing import Union, List, Dict


X_PAD = 0
Y_PAD = -1


class InferYoutubeVideoDataset:
    def __init__(self, img_dir, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, mode="all", transform=None, target_transform=None):
        """
        Generate video clip one by one from start to end for mimic video inference in reality 
        Note that the video frame is sampled by 1 frame/second
        """
        self.max_offset = 2      # positive clip if the distance to GT < 2 offset 
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
        self.asr_files = dict()
        for asr_file in asr_file_list:
            filename = os.path.basename(asr_file)
            vid = filename.split(".")[0][9:]
            self.asr_files[vid] = asr_file

        self.infer_vid = None
        self.transform = transform
        self.target_transform = target_transform
    
    """
    you should run choose_vid before iterate this dataset
    """
    def manual_choose_vid(self, vid):
        if vid in self.vids:
            self.infer_vid = vid
        else:
            raise RuntimeError(f"The vid {vid} is not existed in dataset")
        self._load_gt_data()

    def random_choose_vid(self):
        self.infer_vid = random.sample(self.vids, 1)[0]
        self._load_gt_data()

    def _load_gt_data(self):
        vid = self.infer_vid
        timestamp = self.vid2timestamps[vid]
        asr_file = self.asr_files[vid]
        image_path = os.path.join(self.img_dir, vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))

        with open(asr_file, "r") as f:
            self.subtitles = json.load(f)

        # parse timestamp
        self.cut_points = []         # G.T. cut point + a little bit offset
        self.real_cut_points = []    # record G.T. cut point
        self.descriptions = []
        for timestamp_str in timestamp:
            sec, description = extract_first_timestamp(timestamp_str)
            if sec < 4:
                continue
            if sec > image_num - 4:
                continue
            self.cut_points.append(sec)
            self.real_cut_points.append(sec)
            self.descriptions.append(description)
    

    def __len__(self):
        if self.infer_vid is None:
            raise RuntimeError("You should run choose_vid before iterate this dataset")

        image_path = os.path.join(self.img_dir, self.infer_vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))

        clips = [[start_t, start_t + self.clip_frame_num] for start_t in range(0, image_num - self.clip_frame_num, 2 * self.max_offset)]
        batch_num = len(clips)
        return batch_num
    
    
    def get_duration(self):
        image_path = os.path.join(self.img_dir, self.infer_vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))
        return image_num


    def __getitem__(self, i):
        image_path = os.path.join(self.img_dir, self.infer_vid)
        image_num = len(glob.glob(image_path + "/*.jpg"))
        clips = [[start_t, start_t + self.clip_frame_num] for start_t in range(0, image_num - self.clip_frame_num, 2 * self.max_offset)]

        # this clip's start and end time
        clip_start_sec, clip_end_sec = clips[i]

        # label is determined by IoU
        label = 0
        for cp in self.cut_points:
            pos_st = cp - self.half_clip_frame_num
            pos_et = cp + self.half_clip_frame_num
            a = max(clip_start_sec, pos_st)
            mi = min(clip_start_sec, pos_st)
            b = min(clip_end_sec, pos_et)
            ma = max(clip_end_sec, pos_et)

            iou = (b - a) / (ma - mi) 
            if iou >= (self.clip_frame_num - self.max_offset) / (self.clip_frame_num + self.max_offset):
                label = 1

        # get the subtitle in-between [clip_start_sec - text_extra_time_gap, clip_end_sec + text_extra_time_gap]
        text_extra_time_gap = 1
        text_clip = ""
        for sub in self.subtitles:
            text = sub["text"]
            start_sec = sub["start"]
            if clip_start_sec - text_extra_time_gap < start_sec < clip_end_sec + text_extra_time_gap:
                if len(text_clip) == 0:
                    text_clip += text
                else:
                    text_clip += " " + text
        
        # convert text to ids
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
            img_clip = torch.stack(img_list, dim=0)

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

        return img_clip, text_ids, attention_mask, label


class InferYoutubeClipDataset:
    def __init__(self, img_dir, json_paths, tokenizer, clip_frame_num, max_text_len, mode="all", transform=None, target_transform=None):
        """
        Flat all video data to clips, so that we can testing quickly
        ** Note that you need to run flat_video2clip_for_quick_infer.py in video_chapter_youtube_dataset project to generate test_clips_json file
        """
        self.max_offset = 2      # positive clip if the distance to GT < 2 offset
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.mode = mode        # text (text only), image (image only) or all (multiple-model)
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        self.img_dir = img_dir

        if isinstance(json_paths, list):
            self.all_clip_infos = []
            for file_path in json_paths:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    self.all_clip_infos.extend(data)
        elif isinstance(json_paths, str):
            with open(json_paths, "r") as f:
                self.all_clip_infos = json.load(f)

        # # select specific vid for debug
        # debug_data = []
        # for clip_info in self.all_clip_infos:
        #     vid = clip_info["vid"]
        #     if vid == "ux0EEaaGTR8":
        #         debug_data.append(clip_info)
        # self.all_clip_infos = debug_data

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.all_clip_infos)
        # return len(self.all_clip_infos) // 100

    def __getitem__(self, i):
        clip_info = self.all_clip_infos[i]

        image_paths = clip_info["image_paths"]
        text_clip = clip_info["text_clip"]
        label = clip_info["clip_label"]
        cut_points = clip_info["cut_points"]
        clip_start_sec, clip_end_sec = clip_info["clip_start_end"]
        # vid = clip_info["vid"]

        # convert text to ids
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

        text_ids = torch.from_numpy(np.array(ids)).long()
        attention_mask = torch.from_numpy(np.array(attention_mask)).long()

        if self.mode == "text":
            img_clip = 0    # dummy image
        else:
            # get images in this clip
            img_list = []
            for image_path in image_paths:
                img = Image.open(image_path).convert('RGB')
                img = self.transform(img)
                img_list.append(img)
            img_clip = torch.stack(img_list, dim=0)

            # visualize this clip
            # print(f"https://www.youtube.com/watch?v={vid}&t={t}")
            # print(f"{clip_start_sec} - {clip_end_sec}")
            # print(text_clip)
            # print(label)

            # h, w, c = img.shape
            # clip_whole_image = np.zeros((h, wlen(img_list), c), dtype=np.uint8)
            # for idx, im in enumerate(img_list):
            #     clip_whole_image[:, idxw:(idx+1)*w, :] = im
            # im = Image.fromarray(clip_whole_image)
            # im.save("./clip.jpg")
            # # plt.imshow(clip_whole_image)
            # # plt.show()

        return img_clip, text_ids, attention_mask, label

class InferYoutubeAllClipDataset:
    def __init__(self, img_dir, json_paths, tokenizer, clip_frame_num, max_text_len, mode="all", transform=None, target_transform=None):
        """
        Flat all video data to clips for testing
        Returns all clips' information for a video along with the target clip
        """
        self.max_offset = 2
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.mode = mode
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        self.img_dir = img_dir

        # Load clip infos from json
        if isinstance(json_paths, list):
            all_clips = []
            for file_path in json_paths:
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_clips.extend(json.load(file))
        else:
            with open(json_paths, "r", encoding='utf-8') as f:
                all_clips = json.load(f)

        # Group clips by video
        self.vid_to_clips = {}
        for clip in all_clips:
            vid = clip["vid"]
            if vid not in self.vid_to_clips:
                self.vid_to_clips[vid] = []
            self.vid_to_clips[vid].append(clip)

        # Create an index mapping for iteration
        self.sample_indices = []  # [(vid, clip_index), ...]
        for vid, clips in self.vid_to_clips.items():
            for i in range(len(clips)):
                self.sample_indices.append((vid, i))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, i):
        vid, target_idx = self.sample_indices[i]
        vid_clips = self.vid_to_clips[vid]

        # Process all clips for the current video
        all_clip_images = []
        all_text_ids = []
        all_attention_masks = []

        # Process each clip in the video
        for clip_info in vid_clips:
            # Process images
            if self.mode != "text":
                clip_imgs = []
                for image_path in clip_info["image_paths"]:
                    img = Image.open(image_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    clip_imgs.append(img)
                all_clip_images.append(torch.stack(clip_imgs))

            # Process text
            text_clip = "[CLS] " + clip_info["text_clip"]
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
            img_clips = torch.tensor(0)  # dummy image
        else:
            img_clips = torch.stack(all_clip_images)  # [num_clips, num_frames, C, H, W]
            
        text_ids = torch.stack(all_text_ids)          # [num_clips, max_text_len]
        attention_masks = torch.stack(all_attention_masks)  # [num_clips, max_text_len]
        
        # Get label for target clip
        label = torch.tensor(vid_clips[target_idx]["clip_label"])

        # print(f'img_clips: {img_clips.shape}')
        # print(f'text_ids: {text_ids.shape}')
        # print(f'attention_masks: {attention_masks.shape}')
        # print(f'label: {label}')
        # print(f'target_idx: {target_idx}')

        return img_clips, text_ids, attention_masks, label, torch.tensor(target_idx)
        
    def get_clip_info(self, index):
        """
        Get detailed information about the clip and its video
        """
        vid, target_idx = self.sample_indices[index]
        vid_clips = self.vid_to_clips[vid]
        return {
            "vid": vid,
            "total_clips": len(vid_clips),
            "target_clip": vid_clips[target_idx],
            "all_clips": vid_clips
        }

class InferWindowClipDataset:
    def __init__(self, img_dir, json_paths, tokenizer, clip_frame_num, max_text_len, window_size=2, mode="all", transform=None):
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.window_size = window_size
        self.mode = mode
        self.img_dir = img_dir
        self.transform = transform

        # Load clip infos from json file
        self.all_clip_infos = []
        if isinstance(json_paths, str):
            json_paths = [json_paths]
            
        for json_path in json_paths:
            with open(json_path, "r") as f:
                clip_infos = json.load(f)
                self.all_clip_infos.extend(clip_infos)

        # Group clips by video ID
        self.vid2clips = {}
        for idx, clip_info in enumerate(self.all_clip_infos):
            vid = clip_info["vid"]
            if vid not in self.vid2clips:
                self.vid2clips[vid] = []
            self.vid2clips[vid].append(idx)

    def get_clip_info(self, idx):
        """Get info for a single clip"""
        clip_info = self.all_clip_infos[idx]
        vid = clip_info["vid"]
        clip_start_sec, clip_end_sec = clip_info["clip_start_end"]
        
        # Get video-level indices
        vid_clip_indices = self.vid2clips[vid]
        current_pos = vid_clip_indices.index(idx)
        
        # Get window clip indices
        window_indices = []
        for pos in range(current_pos - self.window_size, current_pos + self.window_size + 1):
            if 0 <= pos < len(vid_clip_indices):
                window_indices.append(vid_clip_indices[pos])
            else:
                window_indices.append(-1)  # Padding index
        
        return clip_info, window_indices

    def __getitem__(self, i):
        clip_info, window_indices = self.get_clip_info(i)
        vid = clip_info["vid"]
        image_path = os.path.join(self.img_dir, vid)

        all_clip_images = []
        all_text_ids = []
        all_attention_masks = []

        for idx in window_indices:
            if idx == -1:
                # Add padding for out-of-bounds indices
                if self.mode != "text":
                    padding_img = torch.zeros((self.clip_frame_num, 3, 224, 224))
                    all_clip_images.append(padding_img)
                
                padding_text = torch.zeros(self.max_text_len, dtype=torch.long)
                padding_mask = torch.zeros(self.max_text_len, dtype=torch.long)
                
                all_text_ids.append(padding_text)
                all_attention_masks.append(padding_mask)
                continue

            window_clip = self.all_clip_infos[idx]
            start_sec, end_sec = window_clip["clip_start_end"]
            subtitle = window_clip["text_clip"]

            # Process images
            if self.mode != "text":
                clip_imgs = []
                for frame_idx in range(start_sec, end_sec):
                    image_filename = f"{frame_idx+1:05d}.jpg"
                    image_path_full = os.path.join(image_path, image_filename)
                    img = Image.open(image_path_full).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    clip_imgs.append(img)
                all_clip_images.append(torch.stack(clip_imgs))

            # Process text
            text_clip = "[CLS] " + subtitle
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
            img_clips = torch.stack(all_clip_images)
            
        text_ids = torch.stack(all_text_ids)
        attention_masks = torch.stack(all_attention_masks)
        label = torch.tensor(clip_info["clip_label"])
        center_idx = self.window_size  # target clip은 항상 중앙에 위치

        return img_clips, text_ids, attention_masks, label, torch.tensor(center_idx)

    def __len__(self):
        return len(self.all_clip_infos)

if __name__ == "__main__":
    # from common_utils import set_random_seed
    # set_random_seed.use_fix_random_seed()
    # img_dir = "D:/youtube_video_frame_minidataset"
    # data_file = "D:/py3_code/video_chapter_youtube_dataset/dataset/test_mini_dataset.csv"
    clip_frame_num = 16
    max_text_len = 100
    img_dir = "/opt/tiger/youtube_video_frame_dataset"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    test_clips_json = f"/opt/tiger/video_chapter_youtube_dataset/dataset/test_clips_clip_frame_num_{clip_frame_num}.json"
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"
    model_type = "bert"
    
    if model_type == "gpt":
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    elif model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        raise RuntimeError(f"Unknown model type {model_type}")
    
    
    test_vision_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset = InferYoutubeVideoDataset(img_dir, data_file, test_vid_file, tokenizer, clip_frame_num=16, max_text_len=100, mode="all", transform=test_vision_preprocess)
    # dataset.random_choose_vid()
    # dataset.manual_choose_vid(vid="Nohke4UXGIM")

    dataset = InferYoutubeClipDataset(img_dir, test_clips_json, tokenizer, clip_frame_num, max_text_len, mode="all", transform=test_vision_preprocess)
    data_loader_params = dict(batch_size=64, shuffle=False, pin_memory=True, num_workers=0)
    data_loader = DataLoader(dataset, **data_loader_params)

    import time
    st = time.time()
    for img_clip, text_ids, attention_mask, label in data_loader:
        print(img_clip.size())
        print(label.size())

    et = time.time()

    print(f"cost time {(et - st) / len(dataset)}  s/batch")
