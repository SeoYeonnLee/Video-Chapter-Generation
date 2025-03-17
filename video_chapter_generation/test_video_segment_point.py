"""
Test for video chapter timepoint generation

TODO: 
output the segment points to a file so that we can use the results to do text summarization.

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
from tqdm import tqdm



if __name__ == "__main__":
    set_random_seed.use_fix_random_seed()
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_mode', default="all", type=str, help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', default="two_stream", type=str, help="bert, r50tsm, r50, two_stream")
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--epoch', default=3000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--head_type', default="mlp", type=str, help="only work on two_stream model")
    parser.add_argument('--data_type', default="all", type=str, help="all, easy, hard, ambiguous")
    args = parser.parse_args()

    # other hyperparameters
    clip_frame_num = args.clip_frame_num
    max_text_len = 100

    # if args.clip_frame_num > 10:
    #     b = 32
    # else:
    #     b = 64
    checkpoint_dir = f"MVCG"#{args.batch_size}"
    ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/chapter_localization/{checkpoint_dir}/checkpoint_15_score_0.4130.pth"
    result_file = f"./test_results/chapter_localization/{checkpoint_dir}_.txt"
    vid2cut_points_file = f"./test_results/chapter_localization/{checkpoint_dir}_vid2cut_points.json"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/dataset/all_in_one_with_subtitle_final.csv"
    test_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/dataset/dataset_fps1/test_clips_clip_frame_num_{clip_frame_num}.json"
    # test_easy_clips_json = f"/opt/tiger/video_chapter_youtube_dataset/dataset/test_easy_clips_clip_frame_num_{clip_frame_num}.json"
    # test_hard_clips_json = f"/opt/tiger/video_chapter_youtube_dataset/dataset/test_hard_clips_clip_frame_num_{clip_frame_num}.json"
    
    train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/dataset/final_train.txt"
    test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/dataset/final_test.txt"
    img_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/youtube_video_frame_dataset"


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
    if args.data_mode == "text":
        model = lang_model
        model.build_chapter_head()
        model = model.to(args.gpu)
    elif args.data_mode == "image":
        model = vision_model
        model.build_chapter_head()
        model = model.to(args.gpu)
    elif args.data_mode == "all":
        lang_base_model = lang_model.base_model
        vision_base_model = vision_model.base_model
        hidden_size = 128
        model = two_stream.TwoStream(lang_base_model, vision_base_model, lang_model.embed_size, vision_model.feature_dim, clip_frame_num, hidden_size)
        model.build_chapter_head(output_size=2, head_type=args.head_type)
        model = model.to(args.gpu)

        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint["epoch"]
        best_result = checkpoint["best_result"]
        model.load_state_dict(checkpoint["model_state_dict"])    

        model = model.eval()
        
        if hasattr(model, 'vision_model'):
            model.vision_model = model.vision_model.eval()
            print("Vision model set to eval mode")
        if hasattr(model, 'lang_model'):
            model.lang_model = model.lang_model.eval()
            print("Language model set to eval mode")
        
        bn_count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                bn_count += 1

        torch.set_grad_enabled(False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")

    else:
        raise RuntimeError(f"Unknown data mode {args.data_mode}")

    # load checkpoint
    # checkpoint = torch.load(ckpt_path)
    # start_epoch = checkpoint["epoch"]
    # print(f'epoch: {start_epoch}')
    # best_result = checkpoint["best_result"]
    # print(f'best_result: {best_result}')
    # model.load_state_dict(checkpoint["model_state_dict"])

    # test on all videos 
    test_vision_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.data_type == "all":
        infer_video_dataset = InferYoutubeClipDataset(img_dir, test_clips_json, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)
    elif args.data_type == "easy":
        infer_video_dataset = InferYoutubeClipDataset(img_dir, test_easy_clips_json, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)
    elif args.data_type == "hard":         
        infer_video_dataset = InferYoutubeClipDataset(img_dir, test_hard_clips_json, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)
    else:
        raise RuntimeError(f"Unknown data_type {args.data_type}")
    infer_video_loader = DataLoader(
        infer_video_dataset,
        shuffle=False,
        pin_memory=False,
        batch_size=args.batch_size,
        num_workers=4,
        persistent_workers=False
    )

    all_pred_label = []
    all_pred_score = []
    batch_i = -1
    pbar = tqdm(infer_video_loader, total=len(infer_video_loader))
    for img_clip, text_ids, attention_mask, label in pbar:
        global_st = time.time()
        batch_i += 1
        # print(f"process {batch_i}/{len(infer_video_loader)}...")

        st = time.time()
        img_clip = img_clip.float().to(args.gpu)
        text_ids = text_ids.to(args.gpu)
        attention_mask = attention_mask.to(args.gpu)   
        label = label.to(args.gpu)
        et = time.time()
        # print(f"cost time1 {et - st}s")

        # frame index
        start_idx = batch_i * args.batch_size
        end_idx = start_idx + img_clip.shape[0]

        # forward the model
        st = time.time()
        with torch.no_grad():
            if args.data_mode == "text":
                binary_logits, binary_prob = model(text_ids, attention_mask)
            elif args.data_mode == "image":
                binary_logits, binary_prob = model(img_clip)
            elif args.data_mode == "all":
                binary_logits, binary_prob = model(img_clip, text_ids, attention_mask)    
            else:
                raise RuntimeError(f"Unknown data mode {args.data_mode}")

            et = time.time()
            # print(f"cost time2 {et - st}s")

            st = time.time()
            topk_scores, topk_labels = binary_logits.data.topk(1, 1, True, True)
            pred_y = topk_labels.squeeze(1).cpu().numpy()
            scores = binary_prob[:, 1].cpu().numpy()

            all_pred_label.extend(list(pred_y))
            all_pred_score.extend(list(scores))

        et = time.time()
        # print(f"cost time3 {et - st}s")
        
        global_et = time.time()
        # print(f"global cost time {global_et - global_st}s")

        del img_clip, text_ids, attention_mask, label
        del binary_logits, binary_prob, topk_scores, topk_labels
        torch.cuda.empty_cache()

    # gather results
    st = time.time()
    for i in range(len(infer_video_dataset)):
        infer_video_dataset.all_clip_infos[i]["pred_score"] = all_pred_score[i]
        infer_video_dataset.all_clip_infos[i]["pred_label"] = all_pred_label[i]
    et = time.time()
    print(f"cost time 3 {et - st}s")

    
    # calculate evaluation metrics
    vid2cut_points = dict()
    auc_list = []
    map_list = []
    precision_list = []
    precision_3_list = []
    precision_5_list = []
    recall_list = []
    recall_3_list = []
    recall_5_list = []
    precision_list_rand = []
    precision_3_list_rand = []
    precision_5_list_rand = []
    recall_list_rand = []
    recall_3_list_rand = []
    recall_5_list_rand = []

    vid = ""
    clip_pred_scores = []
    clip_pred_labels = []
    clip_gt_labels = []
    clip_duration = 0
    gt_cut_points = []
    for clip_info in infer_video_dataset.all_clip_infos:
        if vid != clip_info["vid"]:
            if len(clip_gt_labels) > 0:
                fpr, tpr, thresholds = metrics.roc_curve(clip_gt_labels, clip_pred_scores, pos_label=1)
                test_auc = metrics.auc(fpr, tpr)
                test_m_ap = metrics.average_precision_score(clip_gt_labels, clip_pred_scores)
                auc_list.append(test_auc)
                map_list.append(test_m_ap)

                # convert clip_label to cut_point
                second_gt_cut_points = convert_clip_label2cut_point(clip_gt_labels, clip_frame_num, infer_video_dataset.max_offset)
                second_pred_cut_points = convert_clip_label2cut_point(clip_pred_labels, clip_frame_num, infer_video_dataset.max_offset)
                second_random_cut_points = [random.randint(0, clip_duration - 1) for x in range(len(gt_cut_points))] 
                vid2cut_points[vid] = {"second_gt_cut_points": second_gt_cut_points, "second_pred_cut_points": second_pred_cut_points}

                recall, recall_3, recall_5, precision, precision_3, precision_5 = calculate_pr(second_gt_cut_points, second_pred_cut_points)
                if recall is not None:
                    recall_list.append(recall)
                    recall_3_list.append(recall_3)
                    recall_5_list.append(recall_5)
                if precision is not None:
                    precision_list.append(precision)
                    precision_3_list.append(precision_3)
                    precision_5_list.append(precision_5)
                
                recall, recall_3, recall_5, precision, precision_3, precision_5 = calculate_pr(second_gt_cut_points, second_random_cut_points)
                if recall is not None:
                    recall_list_rand.append(recall)
                    recall_3_list_rand.append(recall_3)
                    recall_5_list_rand.append(recall_5)
                if precision is not None:
                    precision_list_rand.append(precision)
                    precision_3_list_rand.append(precision_3)
                    precision_5_list_rand.append(precision_5)

            # another video, reinit
            vid = clip_info["vid"]
            clip_gt_labels = [clip_info["clip_label"]]
            clip_pred_scores = [clip_info["pred_score"]]
            clip_pred_labels = [clip_info["pred_label"]]
            clip_duration = clip_info["clip_start_end"][1]
            gt_cut_points = [clip_info["cut_points"]]
        
        clip_start_sec, clip_end_sec = clip_info["clip_start_end"]
        clip_gt_labels.append(clip_info["clip_label"])
        clip_pred_scores.append(clip_info["pred_score"])
        clip_pred_labels.append(clip_info["pred_label"])
        clip_duration = clip_info["clip_start_end"][1]
        gt_cut_points = clip_info["cut_points"]

    ##############
    # add last vid
    ##############
    fpr, tpr, thresholds = metrics.roc_curve(clip_gt_labels, clip_pred_scores, pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    test_m_ap = metrics.average_precision_score(clip_gt_labels, clip_pred_scores)
    auc_list.append(test_auc)
    map_list.append(test_m_ap)

    # convert clip_label to cut_point
    second_gt_cut_points = convert_clip_label2cut_point(clip_gt_labels, clip_frame_num, infer_video_dataset.max_offset)
    second_pred_cut_points = convert_clip_label2cut_point(clip_pred_labels, clip_frame_num, infer_video_dataset.max_offset)
    second_random_cut_points = [random.randint(0, clip_duration - 1) for x in range(len(gt_cut_points))] 
    vid2cut_points[vid] = {"second_gt_cut_points": second_gt_cut_points, "second_pred_cut_points": second_pred_cut_points}

    recall, recall_3, recall_5, precision, precision_3, precision_5 = calculate_pr(second_gt_cut_points, second_pred_cut_points)
    if recall is not None:
        recall_list.append(recall)
        recall_3_list.append(recall_3)
        recall_5_list.append(recall_5)
    if precision is not None:
        precision_list.append(precision)
        precision_3_list.append(precision_3)
        precision_5_list.append(precision_5)
    
    recall, recall_3, recall_5, precision, precision_3, precision_5 = calculate_pr(second_gt_cut_points, second_random_cut_points)
    if recall is not None:
        recall_list_rand.append(recall)
        recall_3_list_rand.append(recall_3)
        recall_5_list_rand.append(recall_5)
    if precision is not None:
        precision_list_rand.append(precision)
        precision_3_list_rand.append(precision_3)
        precision_5_list_rand.append(precision_5)
    
    ##############
    # save and calculate results
    ##############
    # save pred cutpoints to file


    if not os.path.exists(os.path.dirname(vid2cut_points_file)):
        os.makedirs(os.path.dirname(vid2cut_points_file))

    with open(vid2cut_points_file, "w") as f:
        json.dump(vid2cut_points, f)
    
    # calculate recall and precision
    avg_recall = sum(recall_list)/len(recall_list)
    avg_recall_3 = sum(recall_3_list)/len(recall_3_list)
    avg_recall_5 = sum(recall_5_list)/len(recall_5_list)
    avg_precision = sum(precision_list)/len(precision_list)
    avg_precision_3 = sum(precision_3_list)/len(precision_3_list)
    avg_precision_5 = sum(precision_5_list)/len(precision_5_list)

    avg_f = 2 * avg_recall * avg_precision / (avg_recall + avg_precision)
    avg_f_3 = 2 * avg_recall_3 * avg_precision_3 / (avg_recall_3 + avg_precision_3)
    avg_f_5 = 2 * avg_recall_5 * avg_precision_5 / (avg_recall_5 + avg_precision_5)
    print(f"mAP {sum(map_list)/len(map_list)}")
    print(f"recall {avg_recall}, recall@3 {avg_recall_3}, recall@5 {avg_recall_5}")
    print(f"precision {avg_precision}, precision@3 {avg_precision_3}, precision@5 {avg_precision_5}")
    print(f"f-score {avg_f}, f-score@3 {avg_f_3}, f-score@5 {avg_f_5}")


    # random guess results
    avg_recall_rand = sum(recall_list_rand)/len(recall_list_rand)
    avg_recall_3_rand = sum(recall_3_list_rand)/len(recall_3_list_rand)
    avg_recall_5_rand = sum(recall_5_list_rand)/len(recall_5_list_rand)
    avg_precision_rand = sum(precision_list_rand)/len(precision_list_rand)
    avg_precision_3_rand = sum(precision_3_list_rand)/len(precision_3_list_rand)
    avg_precision_5_rand = sum(precision_5_list_rand)/len(precision_5_list_rand)

    avg_f_rand = 2 * avg_recall_rand * avg_precision_rand / (avg_recall_rand + avg_precision_rand)
    avg_f_3_rand = 2 * avg_recall_3_rand * avg_precision_3_rand / (avg_recall_3_rand + avg_precision_3_rand)
    avg_f_5_rand = 2 * avg_recall_5_rand * avg_precision_5_rand / (avg_recall_5_rand + avg_precision_5_rand)
    print(f"recall {avg_recall_rand}, recall@3 {avg_recall_3_rand}, recall@5 {avg_recall_5_rand}")
    print(f"precision {avg_precision_rand}, precision@3 {avg_precision_3_rand}, precision@5 {avg_precision_5_rand}")
    print(f"f-score {avg_f_rand}, f-score@3 {avg_f_3_rand}, f-score@5 {avg_f_5_rand}")


    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))

    with open(result_file, "w") as f:
        f.write(f"mAP {sum(map_list)/len(map_list)}\n")
        f.write(f"recall {avg_recall}, recall@3 {avg_recall_3}, recall@5 {avg_recall_5}\n")
        f.write(f"precision {avg_precision}, precision@3 {avg_precision_3}, precision@5 {avg_precision_5}\n")
        f.write(f"f-score {avg_f}, f-score@3 {avg_f_3}, f-score@5 {avg_f_5}\n")
        f.write("\n")
        f.write(f"recall_rand {avg_recall_rand}, recall_rand@3 {avg_recall_3_rand}, recall_rand@5 {avg_recall_5_rand}\n")
        f.write(f"precision_rand {avg_precision_rand}, precision_rand@3 {avg_precision_3_rand}, precision_rand@5 {avg_precision_5_rand}\n")
        f.write(f"f-score_rand {avg_f_rand}, f-score_rand@3 {avg_f_3_rand}, f-score_rand@5 {avg_f_5_rand}\n")


