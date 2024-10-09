"""
CAM Visualization
"""

import os
import time
import cv2
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from data.infer_youtube_video_dataset import InferYoutubeVideoDataset
from common_utils import set_random_seed
from eval_utils.eval_utils import convert_clip_label2cut_point, calculate_pr
from model.lang import bert_hugface
from model.vision import resnet50_tsm
from model.fusion import two_stream


features_blobs = None
def hook_feature(module, input, output):
    global features_blobs
    features_blobs = output.data.cpu().numpy()


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
    model._modules["base_model"].layer4.register_forward_hook(hook_feature)     # get feature map before global pool
    # model.vision_model.layer4.register_forward_hook(hook_feature)     # get feature map before global pool
    
    # dataset transform
    test_vision_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    vision_unnorm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())  # restore original image


    # visualize a video clip by clip
    infer_video_dataset = InferYoutubeVideoDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)
    infer_video_loader = DataLoader(infer_video_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=0)

    training_easy_vids = ["84WcJjSauJs", "OJLk8qNd2O8"]
    testing_easy_vids = ["Nohke4UXGIM", "c1T60EzcV68", "fKOA0WbIvgw"]
    vid = "OJLk8qNd2O8"
    total_video_infer_time = 0
    total_frames = 0
    
    infer_video_dataset.manual_choose_vid(vid=vid)
    duration = infer_video_dataset.get_duration()
    cut_points = infer_video_dataset.cut_points
    total_frames += duration
    print(f"infer video {infer_video_dataset.infer_vid}, duration {duration}")
    
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
        with torch.no_grad():
            if args.data_mode == "text":
                binary_logits, binary_prob = model(text_ids, attention_mask)
            elif args.data_mode == "image":
                binary_logits, binary_prob = model(img_clip)
            elif args.data_mode == "all":
                binary_logits, binary_prob = model(img_clip, text_ids, attention_mask)    
            else:
                raise RuntimeError(f"Unknown data mode {args.ata_mode}")
        fc_weight = model._modules["head"].weight.data.cpu().numpy()

        # record labels along timeline
        cpu_y = list(label.cpu().numpy())
        topk_scores, topk_labels = binary_logits.data.topk(1, 1, True, True)
        pred_y = list(topk_labels.squeeze(1).cpu().numpy())
        gt_labels.extend(cpu_y)
        pred_labels.extend(pred_y)

        t, depth, h, w = features_blobs.shape
        if cpu_y[0] == 1 and pred_y[0] == 1:
            for i in range(t):
                feat = features_blobs[0, :, :, :].reshape(depth, h*w)
                weight = fc_weight[1, i*depth:(i+1)*depth]
                cam = weight.dot(feat)
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                cam_img = cv2.resize(cam_img, (224, 224))

                img = img_clip[0][i]
                img_cpu = np.uint8(vision_unnorm(img).cpu().numpy() * 225)
                img_cpu = np.transpose(img_cpu, (1, 2, 0))

                heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
                result = heatmap * 0.4 + img_cpu * 0.6
                cv2.imwrite(f"{vid}_cam_{i}.jpg", result)
        
            print()
    
    gt_cut_points = convert_clip_label2cut_point(gt_labels, clip_frame_num, infer_video_dataset.max_offset)
    pred_cut_points = convert_clip_label2cut_point(pred_labels, clip_frame_num, infer_video_dataset.max_offset)
    print(f"gt_cut_points {gt_cut_points}")
    print(f"pred_cut_points {pred_cut_points}")

    


