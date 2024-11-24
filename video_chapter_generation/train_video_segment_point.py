"""
Train models (language, vision or both) on youtube dataset
"""

import math
import os
import time
import logging

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from data.youtube_dataset import YoutubeClipDataset, YoutubeAllClipDataset
from data.infer_youtube_video_dataset import InferYoutubeClipDataset
from model.lang import bert_hugface
from model.vision import resnet50_tsm, resnet50
from model.fusion import two_stream
from common_utils import set_random_seed

from sklearn import metrics


logger = logging.getLogger(__name__)

class TrainerConfig:
    # data mode
    data_mode = "all"

    # optimization parameters
    max_epochs = 3000
    start_epoch = 0
    best_result = float('-inf')
    block_size = 50
    batch_size = 8
    learning_rate = 1e-5
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.01  # only applied on matmul weights

    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    lr_decay_type = "cosine"
    warmup_epochs = 200 
    final_epochs = 2500 

    # checkpoint settings
    ckpt_path = None
    num_workers = 8    # for DataLoader

    # tensorboard writer
    tensorboard_writer = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        # self.device = 'cpu'
        # if torch.cuda.is_available():
            # self.device = torch.cuda.current_device()
            # self.model = self.model.to(self.device)
            # self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch, best_result):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        os.makedirs(os.path.dirname(self.config.ckpt_path), exist_ok=True)
        print("saving %s" % self.config.ckpt_path)
        torch.save({"epoch": epoch, "best_result": best_result, "model_state_dict": raw_model.state_dict()}, self.config.ckpt_path)

    def train(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.config)

        best_result = self.config.best_result
        test_result = float('-inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(self.config.start_epoch, self.config.max_epochs):
            self.run_epoch('train', epoch, self.train_dataset)

            if self.test_dataset is not None and epoch % 30 == 0:
                infer_test_result = self.run_epoch("infer_test", epoch, self.test_dataset)
                test_result = infer_test_result

            # supports early stopping based on the test acc, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_result > best_result
            if self.config.ckpt_path is not None and good_model:
                best_result = test_result
                self.save_checkpoint(epoch, best_result)


    def run_epoch(self, split, epoch, dataset):
        is_train = split == 'train'
        self.model.train(is_train)

        if is_train:
            shuffle = True
        else:
            shuffle = False

        losses = []
        
        loader = DataLoader(dataset, shuffle=shuffle, pin_memory=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
        # pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        pbar = tqdm(enumerate(loader), total=len(loader))

        for it, (img_clip, text_ids, attention_mask, label) in pbar:
        # for it, (img_clips, text_ids, attention_masks, label, target_idx) in pbar:
            
        #     print(f'img_clips: {img_clips.shape}')
        #     print(f'text_ids: {text_ids.shape}')
        #     print(f'attention_masks: {attention_masks.shape}')
        #     print(f'label: {label}')
        #     print(f'target_idx: {target_idx}')

            img_clip = img_clip.float().to(self.device)
            text_ids = text_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)   
            label = label.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                if self.config.data_mode == "text":
                    binary_logits, binary_prob = self.model(text_ids, attention_mask)
                elif self.config.data_mode == "image":
                    binary_logits, binary_prob = self.model(img_clip)
                elif self.config.data_mode == "all":
                    binary_logits, binary_prob = self.model(img_clip, text_ids, attention_mask)    
                else:
                    raise RuntimeError(f"Unknown data mode {self.config.data_mode}")
                
                loss = F.cross_entropy(binary_logits, label)

            # record results
            # topk_scores, topk_labels = binary_logits.data.topk(1, 1, True, True)
            # pred_y = topk_labels.squeeze(1)
            cpu_y = list(label.cpu().numpy())
            scores = binary_prob[:, 1]
            scores = scores.detach().cpu().numpy()

            if is_train:
                pos_num = cpu_y.count(1)
                if len(cpu_y) > pos_num > 0:
                    fpr, tpr, thresholds = metrics.roc_curve(cpu_y, scores, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    m_ap = metrics.average_precision_score(cpu_y, scores)
                else:
                    auc = 0
                    m_ap = 0

            if not is_train:
                start_idx = it * self.config.batch_size
                end_idx = start_idx + img_clip.shape[0]

                for i in range(start_idx, end_idx):
                    pred_idx = i - start_idx
                    dataset.all_clip_infos[i]["pred_score"] = scores[pred_idx]

                losses.append(loss.item())
                
            if is_train:
                # backprop and update the parameters
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                # decay the learning rate based on our progress
                if self.config.lr_decay:
                    # self.tokens += (attention_mask > 0).sum()
                    if epoch < self.config.warmup_epochs:
                        # linear warmup
                        lr_mult = max(epoch / self.config.warmup_epochs, 1e-2)
                    else:
                        if epoch < self.config.final_epochs:
                            progress = epoch / self.config.final_epochs
                        else:
                            progress = 1.0
                        # cosine learning rate decay
                        if self.config.lr_decay_type == "cosine":
                            lr_mult = max(0.001, 0.5 * (1.0 + math.cos(math.pi * progress)))

                        # exponential learning rate decay
                        elif self.config.lr_decay_type == "exp":
                            decay_progress_threshold = 1/5
                            if progress < decay_progress_threshold:
                                lr_mult = 1
                            elif decay_progress_threshold < progress < decay_progress_threshold * 2:
                                lr_mult = 0.1
                            elif decay_progress_threshold * 2 < progress < decay_progress_threshold * 3:
                                lr_mult = 0.01
                            else:
                                lr_mult = 0.001
                        else:
                            raise RuntimeError("Unknown learning rate decay type")

                    lr = self.config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.config.learning_rate

                # report progress
                n_iter = epoch * len(loader) + it
                self.config.tensorboard_writer.add_scalar('Train/loss', loss.item(), n_iter)
                if auc:
                    self.config.tensorboard_writer.add_scalar('Train/auc', auc, n_iter)
                    self.config.tensorboard_writer.add_scalar('Train/m_ap', m_ap, n_iter)
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, auc {auc:.5f}, m_ap {m_ap: .5f}, lr {lr:e}")

        if not is_train:
            test_aucs = []
            test_m_aps = []

            vid = ""
            pred_scores = []
            gt_labels = []
            for clip_info in dataset.all_clip_infos:
                if vid != clip_info["vid"]:
                    vid = clip_info["vid"]

                    if len(gt_labels) > 0:
                        fpr, tpr, thresholds = metrics.roc_curve(gt_labels, pred_scores, pos_label=1)
                        test_auc = metrics.auc(fpr, tpr)
                        test_m_ap = metrics.average_precision_score(gt_labels, pred_scores)
                        test_aucs.append(test_auc)
                        test_m_aps.append(test_m_ap)

                    pred_scores = []
                    gt_labels = []

                # clip_start_sec, clip_end_sec = clip_info["clip_start_end"]
                pred_scores.append(clip_info["pred_score"])
                gt_labels.append(clip_info["clip_label"])

            test_loss = float(np.mean(losses))
            test_auc = float(np.mean(test_aucs))
            test_m_ap = float(np.mean(test_m_aps))
            print(f"{split}, loss: {test_loss}, auc {test_auc}, m_ap {test_m_ap}")
            self.config.tensorboard_writer.add_scalar(f'{split}/loss', test_loss, epoch)
            self.config.tensorboard_writer.add_scalar(f'{split}/auc', test_auc, epoch)
            self.config.tensorboard_writer.add_scalar(f'{split}/m_ap', test_m_ap, epoch)
            return test_m_ap



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_mode', default="all", type=str, help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', default="two_stream", type=str, help="bert, r50tsm, two_stream")
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--epoch', default=280, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--head_type', default="mlp", type=str, help="mlp or attn, only work on two_stream model")
    args = parser.parse_args()

    set_random_seed.use_fix_random_seed()
    batch_size = args.batch_size
    clip_frame_num = args.clip_frame_num
    num_workers = 1
    max_text_len = 100
    start_epoch = 0
    best_result = float('-inf')

    vision_pretrain_ckpt_path = f"/home/work/VCG/Video-Chapter-Generation/video_chapter_generation/checkpoint/r50tsm/batch_{batch_size}_lr_decay_cosine_train_test_split/pretrain.pth"
    lang_pretrain_ckpt_path = f"/home/work/VCG/Video-Chapter-Generation/video_chapter_generation/checkpoint/hugface_bert_pretrain/batch_{batch_size}_lr_decay_cosine_train_test_split/pretrain.pth"
    # ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/{args.data_mode}/{args.model_type}_validation/batch_{batch_size}_head_type_{args.head_type}_clip_frame_num_{args.clip_frame_num}/checkpoint.pth"
    ckpt_path = f"/home/work/VCG/Video-Chapter-Generation/video_chapter_generation/checkpoint/test/head_{args.head_type}_batch_{batch_size}/checkpoint.pth"
    img_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/youtube_video_frame_dataset"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_final.csv"
    test_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/validation_clips_clip_frame_num_{clip_frame_num}.json"

    train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/debugging_train.txt"
    test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_validation.txt"
    tensorboard_log = os.path.dirname(ckpt_path)
    tensorboard_writer = SummaryWriter(tensorboard_log)

    # init model
    # lang model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    lang_model = bert_hugface.BertHugface(pretrain_stage=False)
    if os.path.exists(lang_pretrain_ckpt_path):
        lang_model.load_state_dict(torch.load(lang_pretrain_ckpt_path))

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
    
    if os.path.exists(vision_pretrain_ckpt_path):
        vision_model.load_state_dict(torch.load(vision_pretrain_ckpt_path))


    # two stream model
    if args.data_mode == "text":
        model = lang_model
        model.build_chapter_head()
        model = model.to(args.gpu)
    elif args.data_mode == "image":
        model = vision_model
        model.build_chapter_head()
        model = model.to(args.gpu)
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    elif args.data_mode == "all":
        lang_base_model = lang_model.base_model
        vision_base_model = vision_model.base_model
        hidden_size = 128
        model = two_stream.TwoStream(lang_base_model, vision_base_model, lang_model.embed_size, vision_model.feature_dim, clip_frame_num, hidden_size)
        model.build_chapter_head(output_size=2, head_type=args.head_type)
        model = model.to(args.gpu)
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    else:
        raise RuntimeError(f"Unknown data mode {args.data_mode}")
    
    # load pretrained model
    # if os.path.exists(ckpt_path):
    #     checkpoint = torch.load(ckpt_path)
    #     start_epoch = checkpoint["epoch"]
    #     best_result = checkpoint["best_result"]
    #     model.load_state_dict(checkpoint["model_state_dict"])


    # dataset
    train_vision_preprocess = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter()], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_vision_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = YoutubeClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=train_vision_preprocess)
    # train_dataset = YoutubeAllClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=train_vision_preprocess)
    test_dataset = InferYoutubeClipDataset(img_dir, test_clips_json, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(data_mode=args.data_mode, max_epochs=args.epoch,
                        start_epoch=start_epoch, best_result=best_result, batch_size=batch_size, learning_rate=1e-5, block_size=max_text_len,
                        lr_decay_type=args.lr_decay_type, lr_decay=True, warmup_epochs=args.epoch//100, final_epochs=args.epoch//100*90, 
                        num_workers=num_workers, ckpt_path=ckpt_path, tensorboard_writer=tensorboard_writer)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.device = args.gpu
    trainer.train()