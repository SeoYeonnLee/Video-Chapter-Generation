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

import psutil
import gc
from tqdm import tqdm

def clear_memory():
    """메모리 정리"""
    gc.collect()
    torch.cuda.empty_cache()

def print_memory_status():
    """메모리 사용량 출력"""
    # print("\nMemory Status:")
    
    # GPU 메모리
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            # print(f'GPU {i}: Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB')
    
    # System 메모리
    system_memory = psutil.virtual_memory()
    # print(f'System Memory: {system_memory.used/1024**3:.2f}GB/{system_memory.total/1024**3:.2f}GB ({system_memory.percent}%)')



if __name__ == "__main__":
    set_random_seed.use_fix_random_seed()
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_mode', default="all", type=str, help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', default="two_stream", type=str, help="bert, r50tsm, r50, two_stream")
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--head_type', default="mlp", type=str, help="only work on two_stream model")
    parser.add_argument('--data_type', default="all", type=str, help="all, easy, hard, ambiguous")
    args = parser.parse_args()

    # other hyperparameters
    clip_frame_num = args.clip_frame_num
    max_text_len = 100
    
    ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/head_{args.head_type}_batch_{clip_frame_num}/checkpoint.pth"
    result_file = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/test_results/head_{args.head_type}_batch_{clip_frame_num}_.txt"
    vid2cut_points_file = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/test_results/head_{args.head_type}_batch_{clip_frame_num}_vid2cut_points.json"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_final.csv"
    vision_emb_save_dir = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/youtube_video_vision_emb_clip_frame_num_{clip_frame_num}"
    img_dir = "/home/work/capstone/youtube_video_frame_dataset"

    test_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/test_clips_clip_frame_num_{clip_frame_num}.json"
    val_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/validation_clips_clip_frame_num_{clip_frame_num}.json"
    train1_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/train_clips_clip_frame_num_{clip_frame_num}_1.json"
    train2_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/train_clips_clip_frame_num_{clip_frame_num}_2.json"
    train3_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/train_clips_clip_frame_num_{clip_frame_num}_3.json"
    train4_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/train_clips_clip_frame_num_{clip_frame_num}_4.json"
    
    # json_paths = [train2_clips_json, train3_clips_json, train4_clips_json]
    json_paths = [train4_clips_json]

    # train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_train.txt"
    # test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_validation.txt"
    
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
    del checkpoint  # 체크포인트 즉시 제거
    torch.cuda.empty_cache()
    model.eval()

    # test on all videos 
    test_vision_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for idx, json_path in enumerate(json_paths):

        if args.data_type == "all":
            infer_video_dataset = InferYoutubeClipDataset(
                img_dir=img_dir,
                json_paths=json_path,
                tokenizer=tokenizer,
                clip_frame_num=clip_frame_num,
                max_text_len=max_text_len,
                mode=args.data_mode,
                transform=test_vision_preprocess
            )
        else:
            raise RuntimeError(f"Unknown data_type {args.data_type}")

        infer_video_loader = DataLoader(infer_video_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        all_vision_embs = []
        all_pred_label = []
        all_pred_score = []
        batch_i = -1

        # for img_clip, text_ids, attention_mask, label in infer_video_loader: #!!
        for img_clip, text_ids, attention_mask, label in tqdm(infer_video_loader, desc=f"Processing {json_path}"):
            try:
                global_st = time.time()
                batch_i += 1

                # 메모리 상태 체크
                if batch_i % 10 == 0:
                    print_memory_status()

                st = time.time()
                img_clip = img_clip.float().to(args.gpu)
                text_ids = text_ids.to(args.gpu)
                attention_mask = attention_mask.to(args.gpu)   
                label = label.to(args.gpu)
                et = time.time()

                # frame index
                start_idx = batch_i * args.batch_size
                end_idx = start_idx + img_clip.shape[0]

                # forward pass
                st = time.time()
                with torch.no_grad():
                    binary_logits, binary_prob, vision_emb, lang_emb = model(img_clip, text_ids, attention_mask, return_emb=True) 
                    
                    # 즉시 CPU로 이동 및 numpy 변환
                    vision_emb = vision_emb.detach().cpu().numpy()
                    
                    # GPU 메모리에서 제거
                    del binary_logits, binary_prob, lang_emb
                    torch.cuda.empty_cache()
                    
                et = time.time()
                
                # Save embeddings
                st = time.time()
                for i in range(start_idx, end_idx):
                    vid = infer_video_dataset.all_clip_infos[i]["vid"]
                    start_t, end_t = infer_video_dataset.all_clip_infos[i]["clip_start_end"]

                    save_dir = os.path.join(vision_emb_save_dir, vid)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"vision_emb_{start_t}_{end_t}.npy")
                    
                    np.save(save_path, vision_emb[i-start_idx])
                
                # 배치 처리 후 메모리 정리
                del vision_emb, img_clip, text_ids, attention_mask, label
                clear_memory()
                
                et = time.time()
                
                global_et = time.time()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    # 현재 배치 스킵
                    continue
                else:
                    raise e