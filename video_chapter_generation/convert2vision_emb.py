"""
convert the whole dataset to vision embbedding and save to disk

"""


import torch
import time
import json
import os
import random
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics
from transformers import BertTokenizer
from data.infer_youtube_video_dataset import InferYoutubeClipDataset
from common_utils import set_random_seed
from eval_utils.eval_utils import convert_clip_label2cut_point, calculate_pr
from model.lang import bert_hugface
from model.vision import resnet50_tsm, resnet50
from model.fusion import two_stream



if __name__ == "__main__":
    set_random_seed.use_fix_random_seed()
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu', default=6, type=int)
    parser.add_argument('--data_mode', default="all", type=str, help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', default="two_stream", type=str, help="bert, r50tsm, r50, two_stream")
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--epoch', default=3000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--head_type', default="mlp", type=str, help="only work on two_stream model")
    parser.add_argument('--data_type', default="all", type=str, help="all, easy, hard, ambiguous")
    args = parser.parse_args()

    # other hyperparameters
    clip_frame_num = args.clip_frame_num
    max_text_len = 100

    if args.clip_frame_num > 10:
        b = 32
    else:
        b = 64
    checkpoint_dir = f"{args.model_type}_validation/batch_{b}_head_type_{args.head_type}_clip_frame_num_{args.clip_frame_num}"
    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/{args.data_mode}/{checkpoint_dir}/checkpoint.pth"
    result_file = f"./test_results/{args.data_mode}/{checkpoint_dir}_{args.data_type}_.txt"
    vid2cut_points_file = f"./test_results/{args.data_mode}/{checkpoint_dir}_{args.data_type}_vid2cut_points.json"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    all_clips_json = f"/opt/tiger/video_chapter_youtube_dataset/dataset/all_clips_clip_frame_num_{clip_frame_num}.json"
    vision_emb_save_dir = f"/opt/tiger/youtube_video_vision_emb_clip_frame_num_{clip_frame_num}"
    
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"
    img_dir = "/opt/tiger/youtube_video_frame_dataset"


    # init model
    # lang model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    lang_model = bert_hugface.BertHugface(pretrain_stage=False)

    # vision model
    if args.data_mode == "image":
        if args.model_type == "r50tsm":
            vision_model = resnet50_tsm.Resnet50TSM(segments_size=clip_frame_num, shift_div=8, pretrain_stage=False)
        elif args.model_type == "r50":
            vision_model = resnet50.Resnet50(segments_size=clip_frame_num, pretrain_stage=False)
        else:
            raise RuntimeError(f"Unknown model_type {args.model_type}")
    else:
        vision_model = resnet50_tsm.Resnet50TSM(segments_size=clip_frame_num, shift_div=8, pretrain_stage=False)

    # two stream model
    if args.data_mode == "all":
        lang_base_model = lang_model.base_model
        vision_base_model = vision_model.base_model
        hidden_size = 128
        model = two_stream.TwoStream(lang_base_model, vision_base_model, lang_model.embed_size, vision_model.feature_dim, clip_frame_num, hidden_size)
        model.build_chapter_head(output_size=2, head_type=args.head_type)
        model = model.to(args.gpu)
    else:
        raise RuntimeError(f"Unknown data mode {args.data_mode}")

    # load checkpoint
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint["epoch"]
    best_result = checkpoint["best_result"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    # test on all videos 
    test_vision_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.data_type == "all":
        infer_video_dataset = InferYoutubeClipDataset(img_dir, all_clips_json, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)
    else:
        raise RuntimeError(f"Unknown data_type {args.data_type}")
    infer_video_loader = DataLoader(infer_video_dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=8)


    all_vision_embs = []
    all_pred_label = []
    all_pred_score = []
    batch_i = -1
    for img_clip, text_ids, attention_mask, label in infer_video_loader:
        global_st = time.time()
        batch_i += 1
        print(f"process {batch_i}/{len(infer_video_loader)}...")

        st = time.time()
        img_clip = img_clip.float().to(args.gpu)
        text_ids = text_ids.to(args.gpu)
        attention_mask = attention_mask.to(args.gpu)   
        label = label.to(args.gpu)
        et = time.time()
        print(f"cost time1 {et - st}s")

        # frame index
        start_idx = batch_i * args.batch_size
        end_idx = start_idx + img_clip.shape[0]

        # forward the model
        st = time.time()
        with torch.no_grad():
            binary_logits, binary_prob, vision_emb, lang_emb = model(img_clip, text_ids, attention_mask, return_emb=True) 
        et = time.time()
        print(f"cost time2 {et - st}s")
        
        # save vision emb
        st = time.time()
        # vision_emb = torch.mean(vision_emb, dim=1)
        vision_emb = list(vision_emb.detach().cpu().numpy())
        for i in range(start_idx, end_idx):
            vid = infer_video_dataset.all_clip_infos[i]["vid"]
            start_t, end_t = infer_video_dataset.all_clip_infos[i]["clip_start_end"]

            os.makedirs(os.path.join(vision_emb_save_dir, vid), exist_ok=True)
            save_path = os.path.join(vision_emb_save_dir, vid, f"vision_emb_{start_t}_{end_t}.npy")
            with open(save_path, 'wb') as f:
                np.save(f, vision_emb[i-start_idx])
            
        et = time.time()
        print(f"save vision embs cost time {et - st}s")
        
        global_et = time.time()
        print(f"global cost time {global_et - global_st}s")



