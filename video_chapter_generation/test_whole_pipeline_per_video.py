"""
test the whole pipeline (video segment + chapter title generation)

"""

import sys
import os, glob
import json
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics
from transformers import BertTokenizer, PegasusTokenizer
from data.infer_youtube_video_dataset import InferYoutubeVideoDataset
from data.infer_single_video_chapter_title_dataset import InferSingleVideoChapterTitleDataset
from common_utils import set_random_seed
from eval_utils.eval_utils import convert_clip_label2cut_point, calculate_pr
from model.lang import bert_hugface, pegasus_hugface
from model.vision import resnet50_tsm
from model.fusion import two_stream



if __name__ == "__main__":
    training_easy_vids = ["84WcJjSauJs", "DSFt0wHScOA", "OJLk8qNd2O8"]
    testing_easy_vids = ["Nohke4UXGIM", "c1T60EzcV68", "fKOA0WbIvgw", "ux0EEaaGTR8", "YpWIuW1HTos", "z-DxeOt9dsc", "U9DX-eWikZg", "UpTdUk_DV_M"]
    testing_hard_vids = ["xT0J6azBenQ", ]
    test_vid = "xT0J6azBenQ"


    set_random_seed.use_fix_random_seed()
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu', default=6, type=int)
    parser.add_argument('--data_mode', default="all", type=str, help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', default="two_stream", type=str, help="bert, r50tsm, two_stream")
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--epoch', default=3000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--head_type', default="mlp", type=str, help="only work on two_stream model")
    args = parser.parse_args()


    # print to file
    orig_stdout = sys.stdout
    f = open(f"test_results/test_whole_pipeline_result_{test_vid}_clip_frame_num_{args.clip_frame_num}.txt", 'w')
    sys.stdout = f


    # video segment
    # checkpoint_dir = f"{args.model_type}/batch_64_lr_decay_cosine_head_type_{args.head_type}"
    if args.clip_frame_num > 10:
        b = 32
    else:
        b = 64
    checkpoint_dir = f"{args.model_type}/batch_{b}_head_type_{args.head_type}_clip_frame_num_{args.clip_frame_num}"
    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/{args.data_mode}/{checkpoint_dir}/checkpoint.pth"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"
    img_dir = "/opt/tiger/youtube_video_frame_dataset"

    vid_file = test_vid_file

    # other hyperparameters
    max_text_len = 100
    clip_frame_num = args.clip_frame_num

    #### stage 1. vidoe_segment_model  ####
    # lang model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    lang_model = bert_hugface.BertHugface(pretrain_stage=False)

    # vision model
    vision_model = resnet50_tsm.Resnet50TSM(segments_size=clip_frame_num, shift_div=8, pretrain_stage=False)

    # two stream model
    if args.data_mode == "text":
        vidoe_segment_model = lang_model
        vidoe_segment_model.build_chapter_head()
        vidoe_segment_model = vidoe_segment_model.to(args.gpu)
    elif args.data_mode == "image":
        vidoe_segment_model = vision_model
        vidoe_segment_model.build_chapter_head()
        vidoe_segment_model = vidoe_segment_model.to(args.gpu)
    elif args.data_mode == "all":
        lang_base_model = lang_model.base_model
        vision_base_model = vision_model.base_model
        hidden_size = 128
        vidoe_segment_model = two_stream.TwoStream(lang_base_model, vision_base_model, lang_model.embed_size, vision_model.feature_dim, clip_frame_num, hidden_size)
        vidoe_segment_model.build_chapter_head(output_size=2, head_type=args.head_type)
        vidoe_segment_model = vidoe_segment_model.to(args.gpu)
    else:
        raise RuntimeError(f"Unknown data mode {args.data_mode}")

    # load checkpoint for vidoe_segment_model
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint["epoch"]
    best_result = checkpoint["best_result"]
    vidoe_segment_model.load_state_dict(checkpoint["model_state_dict"])
    vidoe_segment_model.eval()

    
    # dataset transform
    test_vision_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    vision_unnorm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())  # restore original image


    # visualize a video clip by clip
    infer_video_dataset = InferYoutubeVideoDataset(img_dir, data_file, vid_file, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)
    infer_video_loader = DataLoader(infer_video_dataset, shuffle=False, pin_memory=True, batch_size=16, num_workers=4)

    infer_video_dataset.manual_choose_vid(vid=test_vid)
    duration = infer_video_dataset.get_duration()
    cut_points = infer_video_dataset.cut_points
    print(f"infer video {infer_video_dataset.infer_vid}, duration {duration}")
    
    t = 0
    batch_i = -1
    time2text = dict() 
    gt_labels = []
    pred_labels = []
    for img_clip, text_ids, attention_mask, label in infer_video_loader:
        batch_i += 1
        print(f"infer video clip {batch_i}/{len(infer_video_loader)}...")

        # original_text = tokenizer.batch_decode(text_ids)

        img_clip = img_clip.float().to(args.gpu)
        text_ids = text_ids.to(args.gpu)
        attention_mask = attention_mask.to(args.gpu)   
        label = label.to(args.gpu)

        # forward the vidoe_segment_model
        with torch.no_grad():
            if args.data_mode == "text":
                binary_logits, binary_prob = vidoe_segment_model(text_ids, attention_mask)
            elif args.data_mode == "image":
                binary_logits, binary_prob = vidoe_segment_model(img_clip)
            elif args.data_mode == "all":
                binary_logits, binary_prob = vidoe_segment_model(img_clip, text_ids, attention_mask)    
            else:
                raise RuntimeError(f"Unknown data mode {args.ata_mode}")

        # record labels along timeline
        cpu_y = list(label.cpu().numpy())
        topk_scores, topk_labels = binary_logits.data.topk(1, 1, True, True)
        pred_y = list(topk_labels.squeeze(1).cpu().numpy())

        gt_labels.extend(cpu_y)
        pred_labels.extend(pred_y)


    # convert clip_label to cut_point
    gt_cut_points = convert_clip_label2cut_point(gt_labels, clip_frame_num, infer_video_dataset.max_offset)
    pred_cut_points = convert_clip_label2cut_point(pred_labels, clip_frame_num, infer_video_dataset.max_offset)
    print(f"gt_cut_points {gt_cut_points}")
    print(f"pred_cut_points {pred_cut_points}")

    # load asr file to get subtitle
    asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
    vid2asr_files = dict()
    for asr_file in asr_file_list:
        filename = os.path.basename(asr_file)
        vid = filename.split(".")[0][9:]
        vid2asr_files[vid] = asr_file
    
    asr_file = vid2asr_files[test_vid]
    with open(asr_file, "r") as f:
        subtitles = json.load(f)
    

    time_gap = 5
    gt_cut_point_texts = []
    gt_timepoint_idx = 0
    clip_text_gt = ""
    for sub in subtitles:
        text = sub["text"]
        start_sec = sub["start"]

        if start_sec > gt_cut_points[gt_timepoint_idx] + time_gap:
            gt_timepoint_idx += 1
            gt_cut_point_texts.append(clip_text_gt)
            clip_text_gt = ""
        if gt_timepoint_idx >= len(gt_cut_points):
            break
        if gt_cut_points[gt_timepoint_idx] - time_gap < start_sec < gt_cut_points[gt_timepoint_idx] + time_gap:
            if len(clip_text_gt) == 0:
                clip_text_gt = text
            else:
                clip_text_gt += " " + text
    
    pred_cut_point_texts = []
    pred_timepoint_idx = 0
    clip_text_pred = ""
    for sub in subtitles:
        text = sub["text"]
        start_sec = sub["start"]

        if start_sec > pred_cut_points[pred_timepoint_idx] + time_gap:
            pred_timepoint_idx += 1
            pred_cut_point_texts.append(clip_text_pred)
            clip_text_pred = ""
        if pred_timepoint_idx >= len(pred_cut_points):
            break
        if pred_cut_points[pred_timepoint_idx] - time_gap < start_sec < pred_cut_points[pred_timepoint_idx] + time_gap:
            if len(clip_text_pred) == 0:
                clip_text_pred = text
            else:
                clip_text_pred += " " + text

    
    print("=== text round gt segment points ===")
    for i, te in enumerate(gt_cut_point_texts):
        print("%d: "%gt_cut_points[i] + te)

    print()
    print("=== text round pred segment points ===")
    for i, te in enumerate(pred_cut_point_texts):
        print("%d: "%pred_cut_points[i] + te)


    #### stage 2. chapter_title_model  ####
    chapter_title_ckpt_path = "/opt/tiger/video_chapter_generation/checkpoint/chapter_title_gen/chapter_title_hugface_pegasus/batch_64_lr_decay_cosine/checkpoint.pth"
    # chapter_title_ckpt_path = "/opt/tiger/video_chapter_generation/checkpoint/chapter_title_hugface_pegasus/batch_64_lr_decay_cosine/checkpoint.pth"

    chapter_title_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
    chapter_title_model = pegasus_hugface.PegasusHugface(reinit_head=True).to(args.gpu)

    checkpoint = torch.load(chapter_title_ckpt_path)
    start_epoch = checkpoint["epoch"]
    best_result = checkpoint["best_result"]
    chapter_title_model.load_state_dict(checkpoint["model_state_dict"])
    # chapter_title_model.load_state_dict(torch.load(chapter_title_ckpt_path))
    chapter_title_model.eval()


    # test chapter title model
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
    infer_single_video_chapter_title_dataset = InferSingleVideoChapterTitleDataset(data_file, vid_file, tokenizer, max_text_len=512)
    data_loader_params = dict(batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
    data_loader = DataLoader(infer_single_video_chapter_title_dataset, **data_loader_params)

    infer_single_video_chapter_title_dataset.manual_choose_vid_and_cut_points(vid=test_vid, cut_points=pred_cut_points)
    # infer_single_video_chapter_title_dataset.manual_choose_vid_and_cut_points(vid=test_vid, cut_points=gt_cut_points)

    gt_titles = infer_single_video_chapter_title_dataset.gt_descriptions
    gen_titles = []
    for text_ids, attention_mask in data_loader:
        original_text = tokenizer.batch_decode(text_ids)[0]
        if len(original_text) <= 0:
            continue
        gen_text, sentence_logits = chapter_title_model.generate(original_text, tokenizer, args.gpu)
        gen_titles.append(gen_text)
    
    print("=== chapter title for gt segment points ===")
    for i, te in enumerate(gt_titles):
        if i == 0:
            print("0: " + te)
        else:
            print("%d: "%gt_cut_points[i - 1] + te)

    print()
    print("=== chapter title for pred segment points ===")
    for i, te in enumerate(gen_titles):
        if i == 0:
            print("0: " + te)
        else:
            print("%d: "%pred_cut_points[i - 1] + te)


    sys.stdout = orig_stdout
    f.close()





