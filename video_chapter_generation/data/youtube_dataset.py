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
        real_cut_points = []    # record G.T. cut point
        descriptions = []
        for timestamp_str in timestamp:
            sec, description = extract_first_timestamp(timestamp_str)
            if sec < 4:
                continue
            if sec > image_num - 4:
                continue
            
            cut_points.append(sec)
            real_cut_points.append(sec)
            descriptions.append(description)

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
            # img_clip = 0

            # visualize this clip
            # print(f"https://www.youtube.com/watch?v={vid}&t={clip_start_sec}")
            # print(f"{clip_start_sec} - {clip_end_sec}")
            # print(text_clip)
            # print(label)

            # h, w = img.size
            # clip_whole_image = np.zeros((h, w*len(img_list), 3), dtype=np.uint8)
            # for idx, im in enumerate(img_list):
            #     clip_whole_image[:, idx*w:(idx+1)*w, :] = im
            # im = Image.fromarray(clip_whole_image)
            # im.save("./clip.jpg")
            # # plt.imshow(clip_whole_image)
            # # plt.show()
            # print()
        
        # visualize text
        # print(f"vid {vid}, label {label}, {clip_start_sec} - {clip_end_sec}, {text_clip}")

        return img_clip, text_ids, attention_mask, label

    def __len__(self):
        return len(self.vids)


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
