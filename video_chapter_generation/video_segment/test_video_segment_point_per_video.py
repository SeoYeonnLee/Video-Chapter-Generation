"""
Test for video chapter timepoint generation per video
Visualization

"""

import os
import time
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics
from transformers import OpenAIGPTTokenizer, BertTokenizer
from data.infer_youtube_video_dataset import InferYoutubeVideoDataset
from common_utils import set_random_seed
from eval_utils.eval_utils import convert_clip_label2cut_point, calculate_pr
from model.lang import bert_hugface
from model.vision import resnet50_tsm
from model.fusion import two_stream


if __name__ == "__main__":
    set_random_seed.use_fix_random_seed()
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu', default=7, type=int)
    parser.add_argument('--data_mode', default="image", type=str, help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', default="r50tsm", type=str, help="bert, r50tsm, two_stream")
    parser.add_argument('--epoch', default=3000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--head_type', default="mlp", type=str, help="only work on two_stream model")
    args = parser.parse_args()


    checkpoint_dir = f"{args.model_type}/batch_32_head_type_{args.head_type}_clip_frame_num_{args.clip_frame_num}"
    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/{args.data_mode}/{checkpoint_dir}/checkpoint.pth"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"
    img_dir = "/opt/tiger/youtube_video_frame_dataset"


    # other hyperparameters
    clip_frame_num = args.clip_frame_num
    max_text_len = 100

    # init model
    # lang model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    lang_model = bert_hugface.BertHugface(pretrain_stage=False)

    # vision model
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
    else:
        raise RuntimeError(f"Unknown data mode {args.data_mode}")

    # load checkpoint
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint["epoch"]
    best_result = checkpoint["best_result"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # dataset transform
    test_vision_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    vision_unnorm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())  # restore original image


    # visualize a video clip by clip
    infer_video_dataset = InferYoutubeVideoDataset(img_dir, data_file, test_vid_file, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)
    infer_video_loader = DataLoader(infer_video_dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=16)

    training_easy_vids = ["84WcJjSauJs"]
    testing_easy_vids = ["Nohke4UXGIM", "c1T60EzcV68", "fKOA0WbIvgw"]
    total_video_infer_time = 0
    total_frames = 0
    for testing_easy_vid in testing_easy_vids:
        infer_video_dataset.manual_choose_vid(vid=testing_easy_vid)
        duration = infer_video_dataset.get_duration()
        cut_points = infer_video_dataset.cut_points
        total_frames += duration
        print(f"infer video {infer_video_dataset.infer_vid}, duration {duration}")
        
        video_infer_time = 0
        t = 0
        batch_i = -1
        gt_labels = []
        pred_labels = []
        for img_clip, text_ids, attention_mask, label in infer_video_loader:
            batch_i += 1

            img_clip = img_clip.float().to(args.gpu)
            text_ids = text_ids.to(args.gpu)
            attention_mask = attention_mask.to(args.gpu)   
            label = label.to(args.gpu)

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
                    raise RuntimeError(f"Unknown data mode {args.ata_mode}")
            et = time.time()
            video_infer_time += et - st

            # record labels along timeline
            cpu_y = list(label.cpu().numpy())
            topk_scores, topk_labels = binary_logits.data.topk(1, 1, True, True)
            pred_y = list(topk_labels.squeeze(1).cpu().numpy())

            gt_labels.extend(cpu_y)
            pred_labels.extend(pred_y)


            # visualize (should set batch_size=1)
            # clip_start_sec = infer_video_dataset.half_clip_frame_num * batch_i
            # clip_end_sec = clip_start_sec + infer_video_dataset.clip_frame_num
            # print(f"process clip {clip_start_sec}s - {clip_end_sec}s")
            # text_ids_cpu = text_ids.squeeze(0).cpu().numpy()
            # sentence = tokenizer._decode(text_ids_cpu)
            # print(sentence)
            # print(f"gt_y {cpu_y}, pred_y {pred_y}")
            
            # if args.data_mode in ["image", "all"]:
            #     img_clip = img_clip[0]
            #     for img_idx in range(img_clip.size(0)):
            #         img = img_clip[img_idx]
            #         img_cpu = vision_unnorm(img).cpu()
            #         img_cpu = transforms.ToPILImage()(img_cpu)
            #         img_cpu.save("visualization_results/temp_clip_images/%d.jpg"%img_idx)

            # print()
        
        gt_cut_points = convert_clip_label2cut_point(gt_labels, clip_frame_num, infer_video_dataset.max_offset)
        pred_cut_points = convert_clip_label2cut_point(pred_labels, clip_frame_num, infer_video_dataset.max_offset)
        print(f"gt_cut_points {gt_cut_points}")
        print(f"pred_cut_points {pred_cut_points}")

        print(f"video infer time {video_infer_time}")
        total_video_infer_time += video_infer_time
    
    print(f"video infer fps {total_frames/total_video_infer_time}")
    print()

    # visulization a bunch of predicted results 
    # test_video_num = 10
    # infer_video_dataset = InferYoutubeVideoDataset(img_dir, data_file, test_vid_file, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)
    # infer_video_loader = DataLoader(infer_video_dataset, shuffle=False, pin_memory=True, batch_size=64, num_workers=0)

    # for v in range(test_video_num):
    #     infer_video_dataset.random_choose_vid()
    #     duration = infer_video_dataset.get_duration()
    #     cut_points = infer_video_dataset.cut_points
    #     print(f"infer video {infer_video_dataset.infer_vid}, duration {duration}")
        
    #     t = 0
    #     batch_i = -1
    #     gt_labels = np.array([0 for x in range(duration)])
    #     pred_labels = np.array([0 for x in range(duration)])
    #     for img_clip, text_ids, attention_mask, label in infer_video_loader:
    #         batch_i += 1

    #         img_clip = img_clip.float().to(args.gpu)
    #         text_ids = text_ids.to(args.gpu)
    #         attention_mask = attention_mask.to(args.gpu)   
    #         label = label.to(args.gpu)

    #         # forward the model
    #         with torch.no_grad():
    #             if args.data_mode == "text":
    #                 binary_logits, binary_prob = model(text_ids, attention_mask)
    #             elif args.data_mode == "image":
    #                 binary_logits, binary_prob = model(img_clip)
    #             elif args.data_mode == "all":
    #                 binary_logits, binary_prob = model(img_clip, text_ids, attention_mask)    
    #             else:
    #                 raise RuntimeError(f"Unknown data mode {args.ata_mode}")

    #         # record labels along timeline
    #         cpu_y = list(label.cpu().numpy())
    #         topk_scores, topk_labels = binary_logits.data.topk(1, 1, True, True)
    #         pred_y = list(topk_labels.squeeze(1).cpu().numpy())

    #         for idx in range(len(cpu_y)):
    #             gt_yy = cpu_y[idx]
    #             pred_yy = pred_y[idx]
    #             clip_start_sec = t
    #             clip_end_sec = t + infer_video_dataset.clip_frame_num
    #             if gt_yy == 1:
    #                 gt_labels[clip_start_sec: clip_end_sec] = gt_yy
    #             if pred_yy == 1:
    #                 pred_labels[clip_start_sec: clip_end_sec] = pred_yy

    #             t = clip_start_sec + infer_video_dataset.half_clip_frame_num

    #     # convert clip_label to cut_point
    #     gt_cut_points = convert_clip_label2cut_point(gt_labels)
    #     pred_cut_points = convert_clip_label2cut_point(pred_labels)
        
    #     # draw
    #     original_gt = np.array([0 for x in range(duration)])
    #     for cp in cut_points:
    #         original_gt[cp] = 1
    #     clip_gt = np.array([0 for x in range(duration)])
    #     for cp in gt_cut_points:
    #         clip_gt[cp] = 1
    #     clip_pred = np.array([0 for x in range(duration)])
    #     for cp in pred_cut_points:
    #         clip_pred[cp] = 1

    #     fig, axs = plt.subplots(3)
    #     fig.tight_layout(pad=2.5)
    #     fig.suptitle(f"vid {infer_video_dataset.infer_vid}")
    #     fig.subplots_adjust(top=0.85)
    #     axs[0].plot(original_gt)
    #     axs[0].set_title("original_gt", fontsize=10)
    #     axs[1].plot(clip_gt)
    #     axs[1].set_title("clip_gt", fontsize=10)
    #     axs[2].plot(clip_pred)
    #     axs[2].set_title("clip_pred", fontsize=10)
    #     plt.savefig(f"visualization_results/{infer_video_dataset.infer_vid}_video_chapter_prediction.jpg")
    #     plt.clf()

