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
        # cut_points = clip_info["cut_points"]
        # clip_start_sec, clip_end_sec = clip_info["clip_start_end"]
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
