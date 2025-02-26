"""
Using youtube subtitle for self-training language model

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
from torch.utils.data import DataLoader
from transformers import OpenAIGPTTokenizer, BertTokenizer
from data.common_utils import parse_csv_to_list, extract_timestamp, load_glove_from_pickle, text_decontracted


X_PAD = 0
Y_PAD = -1

def use_fix_random_seed():
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

class YoutubeClipSubtitleGloveDataset:
    def __init__(self, data_file, glove_pickle_file, vocab_file, clip_frame_num, max_text_len, transform=None, target_transform=None):
        """
        Sample text for SSL training language model by using glove word embedding
        Note that the video frame is sampled by 1 frame/second
        """
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        vids, titles, durations, timestamps = parse_csv_to_list(data_file)
        self.vids = vids
        self.durations = durations
        self.timestamps = timestamps
        # asr files
        asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
        self.asr_files = dict()
        for asr_file in asr_file_list:
            filename = os.path.basename(asr_file)
            vid = filename.split(".")[0][9:]
            self.asr_files[vid] = asr_file

        # load word2vec file
        self.token2embedding = load_glove_from_pickle(glove_pickle_file)
        with open(vocab_file, "r") as f:
            token_list = f.readlines()
            token_list = [x.strip() for x in token_list]
        self.token2id = { ch:i for i,ch in enumerate(token_list) }
        self.id2token = { i:ch for i,ch in enumerate(token_list) }
        self.vocab_size = len(token_list)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        vid = self.vids[i]
        timestamp = self.timestamps[i]
        asr_file = self.asr_files[vid]
        image_num = round(self.durations[i] - 1)    # equal to video total duration (seconds)

        with open(asr_file, "r") as f:
            subtitles = json.load(f)

        # parse timestamp
        cut_points = []
        descriptions = []
        for timestamp_str in timestamp:
            t, sec, si, ei = extract_timestamp(timestamp_str)
            if sec == 0:
                continue
            description = timestamp_str[:si] + timestamp_str[ei:]
            cut_points.append(sec)
            descriptions.append(description)


        t_candidate = list(range(self.half_clip_frame_num, image_num - self.half_clip_frame_num))
        t = random.sample(t_candidate, k=1)[0]
        
        # sampled clip, [start_second, end_second]
        clip_start_sec = t - self.half_clip_frame_num
        clip_end_sec = t + self.half_clip_frame_num

        # get the subtitle in-between [clip_start_sec - text_extra_time_gap, clip_end_sec + text_extra_time_gap]
        text_extra_time_gap = 4 
        text_clip = ""
        for sub in subtitles:
            text = sub["text"]
            start_sec = sub["start"]
            if clip_start_sec - text_extra_time_gap < start_sec < clip_end_sec + text_extra_time_gap:
                if len(text) == 0:
                    text_clip += text
                else:
                    text_clip += " " + text
        
        ids = []
        word_embeddings = []
        text_clip = text_clip.lower()
        text_clip = text_decontracted(text_clip)
        word_str = text_clip.split(" ")
        for w in word_str:
            if len(w) <= 0:
                continue
            if w in self.token2id:   # only process known tokens
                word_embeddings.append(self.token2embedding[w])
                ids.append(self.token2id[w])
        
        x = word_embeddings[:-1]
        y = ids[1:]

        # truncate or pad
        pad_list = [0.0 for x in range(300)]
        x = x[:self.max_text_len]
        y = y[:self.max_text_len]
        x_len = len(x)
        if x_len < self.max_text_len:
            for i in range(self.max_text_len - x_len):
                x.append(pad_list)
                y.append(Y_PAD)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.vids)



"""
A token id dataset for training embedding from scratch instead of using glove embedding 
"""
class YoutubeClipSubtitleDataset:
    def __init__(self, data_file, vocab_file, clip_frame_num, max_text_len, transform=None, target_transform=None):
        """
        Sample text for SSL training language model
        Note that the video frame is sampled by 1 frame/second
        """
        self.clip_frame_num = clip_frame_num
        self.max_text_len = max_text_len
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        vids, titles, durations, timestamps = parse_csv_to_list(data_file)
        self.vids = vids
        self.durations = durations
        self.timestamps = timestamps
        # asr files
        asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
        self.asr_files = dict()
        for asr_file in asr_file_list:
            filename = os.path.basename(asr_file)
            vid = filename.split(".")[0][9:]
            self.asr_files[vid] = asr_file

        # load word2vec file
        with open(vocab_file, "r") as f:
            token_list = f.readlines()
            token_list = [x.strip() for x in token_list]
        self.token2id = { ch:i for i,ch in enumerate(token_list) }
        self.id2token = { i:ch for i,ch in enumerate(token_list) }
        self.vocab_size = len(token_list)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        vid = self.vids[i]
        timestamp = self.timestamps[i]
        asr_file = self.asr_files[vid]
        image_num = round(self.durations[i] - 1)    # equal to video total duration (seconds)

        with open(asr_file, "r") as f:
            subtitles = json.load(f)

        # parse timestamp
        cut_points = []
        descriptions = []
        for timestamp_str in timestamp:
            t, sec, si, ei = extract_timestamp(timestamp_str)
            if sec == 0:
                continue
            description = timestamp_str[:si] + timestamp_str[ei:]
            cut_points.append(sec)
            descriptions.append(description)


        t_candidate = list(range(self.half_clip_frame_num, image_num - self.half_clip_frame_num))
        t = random.sample(t_candidate, k=1)[0]
        
        # sampled clip, [start_second, end_second]
        clip_start_sec = t - self.half_clip_frame_num
        clip_end_sec = t + self.half_clip_frame_num

        # get the subtitle in-between [clip_start_sec - text_extra_time_gap, clip_end_sec + text_extra_time_gap]
        text_extra_time_gap = 4 
        text_clip = ""
        for sub in subtitles:
            text = sub["text"]
            start_sec = sub["start"]
            if clip_start_sec - text_extra_time_gap < start_sec < clip_end_sec + text_extra_time_gap:
                if len(text) == 0:
                    text_clip += text
                else:
                    text_clip += " " + text
        
        ids = []
        text_clip = text_clip.lower()
        text_clip = text_decontracted(text_clip)
        word_str = text_clip.split(" ")
        for w in word_str:
            if len(w) <= 0:
                continue
            if w in self.token2id:   # only process known tokens
                ids.append(self.token2id[w])
        
        x = ids[:-1]
        y = ids[1:]

        # truncate or pad
        x = x[:self.max_text_len]
        y = y[:self.max_text_len]
        x_len = len(x)
        if x_len < self.max_text_len:
            for i in range(self.max_text_len - x_len):
                x.append(X_PAD)
                y.append(Y_PAD)

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.vids)


"""
youtube subtitle dataset for MLM-bert generative-gpt pretraining
"""
class YoutubeClipSubtitleDatasetForHugFace:
    def __init__(self, data_file, vid_file, model_type, tokenizer, clip_frame_num, max_text_len, subtitle_dir=None, transform=None, target_transform=None):
        """
        Sample text for SSL training language model
        Note that the video frame is sampled by 1 frame/second
        """
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.half_clip_frame_num = int(self.clip_frame_num//2)
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

        # asr files
        # asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
        subtitle_path = os.path.dirname(data_file) if subtitle_dir is None else subtitle_dir
        asr_file_list = glob.glob(subtitle_path + "/*/subtitle_*.json")

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
        image_num = round(self.vid2durations[vid] - 1)    # equal to video total duration (seconds)

        with open(asr_file, "r") as f:
            subtitles = json.load(f)

        # parse timestamp
        cut_points = []
        descriptions = []
        for timestamp_str in timestamp:
            t, sec, si, ei = extract_timestamp(timestamp_str)
            if sec == 0:
                continue
            description = timestamp_str[:si] + timestamp_str[ei:]
            cut_points.append(sec)
            descriptions.append(description)

        t_candidate = list(range(self.half_clip_frame_num, image_num - self.half_clip_frame_num))
        t = random.sample(t_candidate, k=1)[0]
        
        # sampled clip, [start_second, end_second]
        clip_start_sec = t - self.half_clip_frame_num
        clip_end_sec = t + self.half_clip_frame_num

        # get the subtitle in-between [clip_start_sec - text_extra_time_gap, clip_end_sec + text_extra_time_gap]
        text_extra_time_gap = 4 
        text_clip = ""
        for sub in subtitles:
            text = sub["text"]
            start_sec = sub["start"]
            if clip_start_sec - text_extra_time_gap < start_sec < clip_end_sec + text_extra_time_gap:
                if len(text_clip) == 0:
                    text_clip += text
                else:
                    text_clip += " " + text
        
        # convert text to ids and targets
        if self.model_type == "gpt":
            ids = self.tokenizer(text_clip).data["input_ids"]
            x = ids[:-1]
            y = ids[1:]
            x = x[:self.max_text_len]
            y = y[:self.max_text_len]
            attention_mask = [1] * len(y)
            if len(x) < self.max_text_len:
                zero_pad_list = [X_PAD] * (self.max_text_len - len(x))
                y_pad_list = [Y_PAD] * (self.max_text_len - len(x))
                x += zero_pad_list
                y += y_pad_list
                attention_mask += zero_pad_list

            x = torch.from_numpy(np.array(x)).long()
            y = torch.from_numpy(np.array(y)).long()
            attention_mask = torch.from_numpy(np.array(attention_mask)).long()

            return x, y, attention_mask

        elif self.model_type == "bert":
            # put [CLS] at the first, so that we can train sentence representation after pretrained
            text_clip = "[CLS] " + text_clip
            tokens = self.tokenizer.tokenize(text_clip)
            # truncate
            tokens = tokens[:self.max_text_len]

            # 15% tokens are masked
            indices = list(range(1, len(tokens)))
            mask_num = round(len(indices) * 0.15)    
            masked_index = random.sample(indices, mask_num)
            # 10% in masked_index not mask
            not_mask_num = round(len(masked_index) * 0.1)
            not_mask_index = random.sample(masked_index, not_mask_num)
            # 10% in masked_index will be replaced by other token
            replace_mask_num = round(len(masked_index) * 0.1)
            replace_mask_index = random.sample(masked_index, replace_mask_num)

            # mask tokens
            masked_idx2token = dict()
            for a in masked_index:
                masked_idx2token[a] = tokens[a]
                tokens[a] = '[MASK]'
            
            # restore some masked tokens
            for not_i in not_mask_index:
                tokens[not_i] = tokens[not_i]

            # replace some masked tokens
            for replace_i in replace_mask_index:
                st = random.sample(tokens, 1)[0]
                tokens[replace_i] = st

            # pad
            attention_mask = [1] * len(tokens)
            if len(tokens) < self.max_text_len:
                zero_pad_list = [0] * (self.max_text_len - len(tokens))
                pad_list = ["[PAD]"] * (self.max_text_len - len(tokens))
                tokens += pad_list
                attention_mask += zero_pad_list

            # Convert token to vocabulary indices
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            y = [Y_PAD] * len(tokens)
            for k, v in masked_idx2token.items():
                idd = self.tokenizer.convert_tokens_to_ids([v])[0]
                y[k] = idd

            x = torch.from_numpy(np.array(ids)).long()
            y = torch.from_numpy(np.array(y)).long()
            attention_mask = torch.from_numpy(np.array(attention_mask)).long()

            return x, y, attention_mask
        else:
            raise RuntimeError(f"Unknown model type {self.model_type}")


    def __len__(self):
        return len(self.vids)



"""
youtube subtitle dataset for shot-level constrastive pretraining
"""
class YoutubeClipConstrastSubtitleDataset:
    def __init__(self, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, neighbor_size=3, transform=None, target_transform=None):
        """
        Shot constrastive unsupervised pretraining
        Note that the video frame is sampled by 1 frame/second
        """
        self.tokenizer = tokenizer
        self.clip_frame_num = clip_frame_num
        self.half_clip_frame_num = int(self.clip_frame_num//2)
        self.max_text_len = max_text_len
        self.neighbor_size = neighbor_size

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
        # vid = "S33RqLPSqIo"
        vid = self.vids[i]
        asr_file = self.vid2asr_files[vid]
        duration = round(self.vid2durations[vid] - 1)    # equal to video total duration (seconds)

        with open(asr_file, "r") as f:
            subtitles = json.load(f)
        
        # sampled clip, [start_second, end_second]
        s_idx = self.neighbor_size * self.clip_frame_num
        e_idx = (self.neighbor_size + 1) * self.clip_frame_num
        t_candidate = list(range(s_idx, duration - e_idx))
        # re-get data
        if len(t_candidate) <= 0:
            ii = random.randint(0, len(self.vids) - 1)
            return self.__getitem__(ii)
        t = random.sample(t_candidate, k=1)[0]
        
        clip_start_sec = t
        clip_end_sec = clip_start_sec + self.clip_frame_num
        neighbor_clip_start_end = []
        for idx in range(self.neighbor_size * 2):
            if idx < self.neighbor_size:    # left side
                s = clip_start_sec - (self.neighbor_size - idx) * self.clip_frame_num
                e = clip_start_sec - (self.neighbor_size - idx - 1) * self.clip_frame_num
            else:                           # right side
                s = clip_end_sec + (idx - self.neighbor_size) * self.clip_frame_num
                e = clip_end_sec + (idx - self.neighbor_size + 1) * self.clip_frame_num
            neighbor_clip_start_end.append([s, e])
        
        all_clip_start_end = neighbor_clip_start_end[:self.neighbor_size] + [[clip_start_sec, clip_end_sec]] + neighbor_clip_start_end[self.neighbor_size:]
        
        # get the subtitle in all clips
        clip_texts = dict()
        clip_idx = 0        
        for sub in subtitles:
            text = sub["text"]
            start_sec = sub["start"]

            if start_sec > all_clip_start_end[clip_idx][1]:
                clip_idx += 1
            if clip_idx >= len(all_clip_start_end):
                break

            s, e = all_clip_start_end[clip_idx]
            if s <= start_sec <= e :
                if clip_idx in clip_texts:
                    clip_texts[clip_idx] += " " + text
                else:
                    clip_texts[clip_idx] = text

        # there are no subtitle within clip in some video (like S33RqLPSqIo 710s-720s)
        # re-get data
        if len(clip_texts) < self.neighbor_size * 2 + 1:
            ii = random.randint(0, len(self.vids) - 1)
            return self.__getitem__(ii)

        clip_ids = []
        clip_attention_masks = []
        for idx in range(self.neighbor_size * 2 + 1):
            text_clip = clip_texts[idx]
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

            ids = torch.from_numpy(np.array(ids)).long()
            attention_mask = torch.from_numpy(np.array(attention_mask)).long()
            clip_ids.append(ids)
            clip_attention_masks.append(attention_mask)
        
        # query
        query_clip_text_id = clip_ids[self.neighbor_size]
        query_att_mask = clip_attention_masks[self.neighbor_size]

        # positive candidates
        pos_candidates_text_id = clip_ids[:self.neighbor_size] + clip_ids[self.neighbor_size+1:]
        pos_candidates_text_id = torch.stack(pos_candidates_text_id, dim=0)
        pos_candidates_att_mask = clip_attention_masks[:self.neighbor_size] + clip_attention_masks[self.neighbor_size+1:]
        pos_candidates_att_mask = torch.stack(pos_candidates_att_mask, dim=0)

        return query_clip_text_id, query_att_mask, pos_candidates_text_id, pos_candidates_att_mask


    def __len__(self):
        return len(self.vids)




if __name__ == "__main__":
    # from common_utils import set_random_seed
    # set_random_seed.use_fix_random_seed()
    # img_dir = "D:/youtube_video_frame_minidataset"
    # data_file = "D:/py3_code/video_chapter_youtube_dataset/dataset/test_mini_dataset.csv"

    img_dir = "/opt/tiger/youtube_video_frame_minidataset"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    glove_pickle_file = "/opt/tiger/video_chapter_generation/glove.840B.300d.pickle"
    vocab_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/vocab.txt"
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"

    use_fix_random_seed()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = YoutubeClipConstrastSubtitleDataset(data_file, train_vid_file, tokenizer, clip_frame_num=10, max_text_len=50, neighbor_size=3)
    data_loader_params = dict(batch_size=64, shuffle=False, pin_memory=True)
    data_loader = DataLoader(dataset, **data_loader_params)

    for query_clip_text_id, query_att_mask, pos_candidates_text_id, pos_candidates_att_mask in data_loader:
        print(query_clip_text_id.size())         # size: (batch, text_len) 
        print(pos_candidates_text_id.size())     # size: (batch, candid_size, text_len)
        print()



    # youtube_clip_subtitle_glove_dataset = YoutubeClipSubtitleGloveDataset(data_file, glove_pickle_file, vocab_file, clip_frame_num=8, max_text_len=200)
    # data_loader_params = dict(batch_size=64, shuffle=False, pin_memory=True)
    # data_loader = DataLoader(youtube_clip_subtitle_glove_dataset, **data_loader_params)

    # for x, y in data_loader:
    #     mask = torch.nonzero(y == -1)
    #     print(x.size())
    #     print(y.size())
    #     print()


    # tokenizer and model
    # model_type = "bert"
    # if model_type == "gpt":
    #     tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    # elif model_type == "bert":
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # else:
    #     raise RuntimeError(f"Unknown model type {model_type}")
    # dataset = YoutubeClipSubtitleDatasetForHugFace(data_file, model_type, tokenizer, clip_frame_num=8, block_size=50)

    # data_loader_params = dict(batch_size=64, shuffle=False, pin_memory=True)
    # data_loader = DataLoader(dataset, **data_loader_params)

    # for x, y, z in data_loader:
    #     print()

    




