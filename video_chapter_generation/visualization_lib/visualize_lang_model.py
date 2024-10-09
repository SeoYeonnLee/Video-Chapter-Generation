"""
Visualize attention map for language model

"""

import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from model.lang import bert_hugface
from data.youtube_dataset import YoutubeClipDataset
from data.infer_youtube_video_dataset import InferYoutubeVideoDataset
from common_utils import set_random_seed
from visualization_lib.lang.integrated_gradient import IntegratedGradient
from IPython.display import display, HTML


if __name__ == "__main__":
    set_random_seed.use_fix_random_seed()
    device = "cuda:3"
    model_type = "bert"
    lr_decay_type = "cosine"
    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/text/bert/batch_32_head_type_mlp_clip_frame_num_16/checkpoint.pth"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    img_dir = "/opt/tiger/youtube_video_frame_dataset"
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"

    # tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = bert_hugface.BertHugface(pretrain_stage=False)
    model.build_chapter_head()
    model = model.to(device)

    # dataset
    data_mode = "text"  # text (text only), image (image only) or all (multiple-model)
    clip_frame_num = 16
    max_text_len = 100

    # load checkpoint
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint["epoch"]
    best_result = checkpoint["best_result"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    # test on video clip infer dataset
    infer_video_dataset = InferYoutubeVideoDataset(img_dir, data_file, test_vid_file, tokenizer, clip_frame_num, max_text_len=max_text_len, mode=data_mode)
    infer_video_loader = DataLoader(infer_video_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=0)

    # gradient based visualization
    integrated_grad_visualization = IntegratedGradient(model, tokenizer, encoder="base_model")

    # run on dataset
    # infer_video_dataset.random_choose_vid()
    infer_video_dataset.manual_choose_vid(vid="Nohke4UXGIM")
    duration = infer_video_dataset.get_duration()
    print(f"infer video {infer_video_dataset.infer_vid}, duration {duration}")
    
    t = 0
    batch_i = -1
    for img_clip, text_ids, attention_mask, label in infer_video_loader:
        batch_i += 1

        img_clip = img_clip.float().to(device)
        text_ids = text_ids.to(device)
        attention_mask = attention_mask.to(device)   
        label = label.to(device)

        instances = integrated_grad_visualization.saliency_interpret((img_clip, text_ids, attention_mask, label))
        coloder_string = integrated_grad_visualization.colorize(instances[0])
        pred_label = instances[0]["pred_label"]
        gt_label = label.item()

        # if pred_label == 1:
        if gt_label == 1:
            display(HTML(coloder_string))
            display(HTML("<br>"))


