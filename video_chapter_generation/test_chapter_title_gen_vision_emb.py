"""
Test chapter title generation on the whole of dataset.
Calculate the evaluation metrics

"""

import time
import math
import random
import os
import logging

from tqdm import tqdm
from rouge import Rouge 
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import PegasusTokenizer
from data.youtube_chapter_title_dataset import YoutubeAllChapterTitleDataset, YoutubeAllChapterTitlePredictDataset
from model.lang import pegasus_vision_emb
from common_utils import set_random_seed



if __name__ == "__main__":
    set_random_seed.use_fix_random_seed()

    import argparse
    parser = argparse.ArgumentParser(description='video chapter title generation model')
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--data_mode', default="all", type=str)     # easy, hard, all
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--model_type', default="pegasus", type=str)    # pegasus or bigbird
    parser.add_argument('--location_type', default="gt", type=str)    # gt or pred
    parser.add_argument('--fusion_type', default="cross_attn", type=str)    # gt or pred
    args = parser.parse_args()

    vision_emb_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/youtube_video_vision_emb_clip_frame_num_16"
    checkpoint_dir = f"{args.model_type}_batch_{args.batch_size}"
    ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/chapter_title_gen_vision_emb/{checkpoint_dir}/checkpoint-50/checkpoint_50.pth"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_final.csv"
    train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_train.txt"
    test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_test.txt"
    # test_easy_vid_file = f"/opt/tiger/video_chapter_youtube_dataset/dataset/easy_test_vid.txt"
    # test_hard_vid_file = f"/opt/tiger/video_chapter_youtube_dataset/dataset/hard_test_vid.txt"
    # result_file = f"./test_results/chapter_title_gen/{checkpoint_dir}_{args.data_mode}_vid.txt"

    # for title summarization based on predicted cut points
    vid2cut_points_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/test_results/head_mlp_batch_16_vid2cut_points.json"
    if args.location_type == "gt":
        result_file = f"./test_results/chapter_title_gen_vision_emb/{checkpoint_dir}_50.txt"
    else:
        result_file = f"./test_results/chapter_title_gen_vision_emb/{checkpoint_dir}_vid_pred_cut_points.txt"

    # other hyperparameters
    num_workers = 8
    max_text_len = 512
    chapter_title_text_len = 30

    # tokenizer and model
    if args.model_type == "pegasus":
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
        model = pegasus_vision_emb.PegasusVisionEmb(reinit_head=True, fusion_type=args.fusion_type).to(args.gpu)
    else:
        raise RuntimeError(f"Unknown model_type {args.model_type}")

    # load checkpoint
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint["epoch"]
    best_result = checkpoint["best_result"]
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.load_state_dict(checkpoint)
    model.eval()

    # dataset
    if args.data_mode == "all":
        if args.location_type == "gt":
            dataset = YoutubeAllChapterTitleDataset(data_file, test_vid_file, tokenizer, max_text_len, chapter_title_text_len, vision_emb_dir)
        else:
            dataset = YoutubeAllChapterTitlePredictDataset(vid2cut_points_file, data_file, test_vid_file, tokenizer, max_text_len, chapter_title_text_len, vision_emb_dir)
    elif args.data_mode == "easy":
        if args.location_type == "gt":
            dataset = YoutubeAllChapterTitleDataset(data_file, test_easy_vid_file, tokenizer, max_text_len, chapter_title_text_len, vision_emb_dir)
        else:
            dataset = YoutubeAllChapterTitlePredictDataset(vid2cut_points_file, data_file, test_easy_vid_file, tokenizer, max_text_len, chapter_title_text_len, vision_emb_dir)
    elif args.data_mode == "hard":
        if args.location_type == "gt":
            dataset = YoutubeAllChapterTitleDataset(data_file, test_hard_vid_file, tokenizer, max_text_len, chapter_title_text_len, vision_emb_dir)
        else:
            dataset = YoutubeAllChapterTitlePredictDataset(vid2cut_points_file, data_file, test_hard_vid_file, tokenizer, max_text_len, chapter_title_text_len, vision_emb_dir)
    else:
        raise RuntimeError(f"Unknown data mode {args.data_mode}")
    infer_loader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=num_workers)

    losses = []
    accs = []
    gt_texts = []
    gen_texts = []
    original_texts = []
    simple_titles = []
    random_titles = []
    principal_titles = []
    rouge = Rouge()

    batch_i = -1
    for vision_embs, vision_attention_mask, text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids in infer_loader:
        global_st = time.time()
        batch_i += 1
        print(f"process {batch_i}/{len(infer_loader)}...")

        original_batch_text = tokenizer.batch_decode(text_ids)
        original_texts.extend(original_batch_text)
        for oi, ot in enumerate(original_batch_text):
            # lead
            ots = " ".join(ot.split(" ")[:10])
            simple_titles.append(ots)

            # random
            word_list = ot.split(" ")
            sentence_candidates = [" ".join(word_list[k:(k+10)]) for k in range(0, len(word_list), 10)]
            rand_sen = random.sample(sentence_candidates, 1)[0]
            random_titles.append(rand_sen)

            # principal
            word_list = ot.split(" ")
            sentence_candidates = [" ".join(word_list[k:(k+10)]) for k in range(0, len(word_list), 10)]
            scores = []
            for sen in sentence_candidates:
                if len(sen) <= 0:
                    scores.append(0.0)
                    continue
                s = rouge.get_scores(sen, original_batch_text[oi])
                r1_f = s[0]["rouge-1"]["f"]
                scores.append(r1_f)
            idx = scores.index(max(scores))
            selected_sen = sentence_candidates[idx]
            principal_titles.append(selected_sen)

        st = time.time()
        vision_embs = vision_embs.to(args.gpu)
        vision_attention_mask = vision_attention_mask.to(args.gpu)
        text_ids = text_ids.to(args.gpu)
        attention_mask = attention_mask.to(args.gpu)   
        input_decode_ids = input_decode_ids.to(args.gpu)
        decode_attention_mask = decode_attention_mask.to(args.gpu)
        target_decode_ids = target_decode_ids.to(args.gpu)
        et = time.time()
        print(f"cost time1 {et - st}s")

        # forward the model
        gen_batch_text = []
        st = time.time()
        with torch.no_grad():
            logits = model(vision_embs, vision_attention_mask, text_ids, attention_mask, decoder_input_ids=input_decode_ids, decoder_attention_mask=decode_attention_mask)
            for ti, text_id in enumerate(text_ids):
                original_text = tokenizer.decode(text_id.cpu())
                vision_embs_single = vision_embs[ti].unsqueeze(0)
                vision_attention_mask_single = vision_attention_mask[ti].unsqueeze(0)
                if len(original_text) <= 0:
                    gen_batch_text.append("")
                else:
                    gen_text, sentence_logits = model.generate(vision_embs_single, vision_attention_mask_single, original_text, tokenizer, args.gpu)
                    gen_batch_text.append(gen_text)
        et = time.time()
        print(f"cost time2 {et - st}s")


        st = time.time()
        mask = torch.nonzero(decode_attention_mask == 1)
        valid_targets = target_decode_ids[mask[:, 0], mask[:, 1]]
        valid_logits = logits[mask[:, 0], mask[:, 1], :]
        loss = F.cross_entropy(valid_logits.view(-1, valid_logits.size(-1)), valid_targets.view(-1))
        
        cpu_y = valid_targets.cpu().numpy()
        topk_scores, topk_labels = valid_logits.data.topk(1, 1, True, True)
        topk_ind = topk_labels.squeeze(1).cpu().numpy()
        correct = np.sum(topk_ind == cpu_y)
        count = len(cpu_y)
        acc = correct / count

        losses.append(loss.item())
        accs.append(acc)

        # text
        cpu_y = target_decode_ids.cpu().numpy()
        gt_batch_tokens = [tokenizer.convert_ids_to_tokens(x) for x in cpu_y]
        gt_batch_text = [tokenizer.convert_tokens_to_string(x) for x in gt_batch_tokens]

        gt_texts.extend(gt_batch_text)
        gen_texts.extend(gen_batch_text)

        et = time.time()
        print(f"cost time3 {et - st}s")
        
        global_et = time.time()
        print(f"global cost time {global_et - global_st}s")

    
    # calculate evaluation metrics
    test_loss = float(np.mean(losses))
    test_acc = float(np.mean(accs))
    
    # filter empty text
    random_empty_idxs = []
    for i, text in enumerate(random_titles):
        if len(text) == 0:
            random_empty_idxs.append(i)
    
    simple_empty_idxs = []
    for i, text in enumerate(simple_titles):
        if len(text) == 0:
            simple_empty_idxs.append(i)

    principal_empty_idxs = []
    for i, text in enumerate(principal_titles):
        if len(text) == 0:
            principal_empty_idxs.append(i)

    gen_empty_idxs = []
    for i, text in enumerate(gen_texts):
        if len(text) == 0:
            gen_empty_idxs.append(i)

    # see https://pypi.org/project/rouge/
    rouge = Rouge()
    random_gt_texts = [x for i, x in enumerate(gt_texts) if i not in random_empty_idxs]
    random_titles = [x for i, x in enumerate(random_titles) if i not in random_empty_idxs]
    scores = rouge.get_scores(random_titles, random_gt_texts, avg=True)
    rouge1_rand = scores["rouge-1"]
    rouge2_rand = scores["rouge-2"]
    rougeL_rand = scores["rouge-l"]
    print(f"random rouge-1 {rouge1_rand}")
    print(f"random rouge-2 {rouge2_rand}")
    print(f"random rouge-L {rougeL_rand}")

    simple_gt_texts = [x for i, x in enumerate(gt_texts) if i not in simple_empty_idxs]
    simple_titles = [x for i, x in enumerate(simple_titles) if i not in simple_empty_idxs]
    scores = rouge.get_scores(simple_titles, simple_gt_texts, avg=True)
    rouge1_sim = scores["rouge-1"]
    rouge2_sim = scores["rouge-2"]
    rougeL_sim = scores["rouge-l"]
    print(f"lead rouge-1 {rouge1_sim}")
    print(f"lead rouge-2 {rouge2_sim}")
    print(f"lead rouge-L {rougeL_sim}")

    principal_gt_texts = [x for i, x in enumerate(gt_texts) if i not in principal_empty_idxs]
    principal_titles = [x for i, x in enumerate(principal_titles) if i not in principal_empty_idxs]
    scores = rouge.get_scores(principal_titles, principal_gt_texts, avg=True)
    rouge1_pri = scores["rouge-1"]
    rouge2_pri = scores["rouge-2"]
    rougeL_pri = scores["rouge-l"]
    print(f"principal rouge-1 {rouge1_pri}")
    print(f"principal rouge-2 {rouge2_pri}")
    print(f"principal rouge-L {rougeL_pri}")

    gen_gt_texts = [x for i, x in enumerate(gt_texts) if i not in gen_empty_idxs]
    gen_texts = [x for i, x in enumerate(gen_texts) if i not in gen_empty_idxs]
    scores = rouge.get_scores(gen_texts, gen_gt_texts, avg=True)
    rouge1 = scores["rouge-1"]
    rouge2 = scores["rouge-2"]
    rougeL = scores["rouge-l"]

    print(f"test_loss {test_loss}")
    print(f"test_acc {test_acc}")
    print(f"rouge-1 {rouge1}")
    print(f"rouge-2 {rouge2}")
    print(f"rouge-L {rougeL}")
    

    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))

    with open(result_file, "w") as f:
        f.write(f"random rouge-1 {rouge1_rand}\n")
        f.write(f"random rouge-2 {rouge2_rand}\n")
        f.write(f"random rouge-l {rougeL_rand}\n")

        f.write(f"lead rouge-1 {rouge1_sim}\n")
        f.write(f"lead rouge-2 {rouge2_sim}\n")
        f.write(f"lead rouge-l {rougeL_sim}\n")

        f.write(f"principal rouge-1 {rouge1_pri}\n")
        f.write(f"principal rouge-2 {rouge2_pri}\n")
        f.write(f"principal rouge-l {rougeL_pri}\n")

        f.write(f"test_loss {test_loss}\n")
        f.write(f"test_acc {test_acc}\n")
        f.write(f"rouge-1 {rouge1}\n")
        f.write(f"rouge-2 {rouge2}\n")
        f.write(f"rouge-l {rougeL}\n")





