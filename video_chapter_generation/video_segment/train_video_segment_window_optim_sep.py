"""
Train models (language, vision or both) on youtube dataset with global attention
"""

import math
import os
import time
import logging
import gc

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, get_cosine_schedule_with_warmup

from data.youtube_dataset import YoutubeAllClipDataset, WindowClipDataset
from data.infer_youtube_video_dataset import InferYoutubeAllClipDataset, InferWindowClipDataset
from model.lang import bert_hugface
from model.vision import resnet50_tsm
from model.fusion import two_stream_window_optim_sep
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
    batch_size = 16
    learning_rate = 1e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.01

    # Training schedule
    warmup_ratio = 0.1
    validation_interval = 10

    # Misc
    ckpt_path = None
    num_workers = 8
    tensorboard_writer = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.config)

        num_training_steps = len(train_dataset) // config.batch_size * config.max_epochs
        warmup_steps = int(num_training_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

    def save_checkpoint(self, epoch, best_result):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint_dir = os.path.dirname(self.config.ckpt_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create new checkpoint filename with epoch number
        base_path = os.path.splitext(self.config.ckpt_path)[0]  # Remove .pth extension
        new_checkpoint_path = f"{base_path}_{epoch}.pth"
        
        print(f"Saving new best checkpoint at epoch {epoch} to {new_checkpoint_path}")
        
        checkpoint = {
            "epoch": epoch,
            "best_result": best_result,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }
        
        # Save new checkpoint with epoch number
        torch.save(checkpoint, new_checkpoint_path)
        torch.save(checkpoint, self.config.ckpt_path)

    def calculate_metrics(self, labels, scores):
       """Calculate AUC and MAP metrics"""
       if len(labels) > 0:
           fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
           auc = metrics.auc(fpr, tpr)
           m_ap = metrics.average_precision_score(labels, scores)
           return auc, m_ap
       return 0, 0

    def log_metrics(self, split, metrics_dict, step):
       """Log metrics to tensorboard"""
       if hasattr(self.config, 'tensorboard_writer') and self.config.tensorboard_writer is not None:
           for name, value in metrics_dict.items():
               self.config.tensorboard_writer.add_scalar(f'{split}/{name}', value, step)

    def _cleanup_resources(self):
        """Unified resource cleanup"""
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Tensorboard cleanup
        if hasattr(self.config, 'tensorboard_writer') and self.config.tensorboard_writer is not None:
            self.config.tensorboard_writer.flush()

    def train(self):
        best_result = self.config.best_result
        test_result = float('-inf')

        for epoch in range(self.config.start_epoch+1, self.config.max_epochs+1):
            # Training
            self.run_epoch('train', epoch, self.train_dataset)
            
            if epoch != 0:
                if self.test_dataset is not None and epoch % self.config.validation_interval == 0:
                    test_result = self.run_epoch("infer_test", epoch, self.test_dataset)

            if epoch % 5 == 0:
                self._cleanup_resources()
            
            good_model = self.test_dataset is None or test_result > best_result
            if self.config.ckpt_path is not None and good_model:
                best_result = test_result
                self.save_checkpoint(epoch, best_result)
    
    def run_epoch(self, split, epoch, dataset):
        is_train = split == 'train'
        self.model.train(is_train)

        loader = DataLoader(
            dataset,
            shuffle=is_train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader))

        device = next(self.model.parameters()).device

        for it, (img_clips, text_ids, attention_masks, labels, clip_info) in pbar:
            # Move data to device
            img_clips = img_clips.float().to(device)
            text_ids = text_ids.to(device)
            attention_masks = attention_masks.to(device)   
            labels = labels.to(device)

            # Move clip_info tensors to device
            clip_info = {
                'clip_start_frame': clip_info['clip_start_frame'].to(device),
                'total_frames': clip_info['total_frames'].to(device),
                'target_clip_idx': clip_info['target_clip_idx'].to(device),
                'total_num_clips': clip_info['total_num_clips'].to(device)
            }

            # Forward pass
            with torch.set_grad_enabled(is_train):
                # with torch.cuda.amp.autocast():  # Mixed precision
                binary_logits, binary_prob = self.model(img_clips, text_ids, attention_masks, clip_info)
                loss = F.cross_entropy(binary_logits, labels)

            if is_train:              
                # Training metrics
                scores = binary_prob[:, 1].detach().cpu().numpy()
                batch_labels = labels.cpu().numpy()
                batch_auc, batch_m_ap = self.calculate_metrics(batch_labels, scores)

                # Optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()
                self.scheduler.step()

                # Logging
                n_iter = epoch * len(loader) + it
                current_lr = self.scheduler.get_last_lr()[0]

                metrics_dict = {
                    'lr': current_lr,
                    'loss': loss.item(),
                    'batch_auc': batch_auc,
                    'batch_map': batch_m_ap
                }
                self.log_metrics('Train', metrics_dict, n_iter)

                pbar.set_description(
                    f"epoch {epoch} iter {it}: train loss {loss.item():.5f}, "
                    f"auc {batch_auc:.5f}, map {batch_m_ap:.5f}, lr {current_lr:e}"
                )
            
            else:
                scores = binary_prob[:, 1].detach().cpu().numpy()
                start_idx = it * self.config.batch_size
                end_idx = start_idx + img_clips.shape[0]

                for i in range(start_idx, end_idx):
                    pred_idx = i - start_idx
                    dataset.all_clip_infos[i]["pred_score"] = scores[pred_idx]

                losses.append(loss.item())

        if not is_train:
            test_aucs = []
            test_m_aps = []
            vid = ""
            pred_scores = []
            gt_labels = []

            # Collect scores by video
            for clip_info in dataset.all_clip_infos:
                if vid != clip_info["vid"]:
                    if vid != "":  # Process previous video
                        auc, m_ap = self.calculate_metrics(gt_labels, pred_scores)
                        test_aucs.append(auc)
                        test_m_aps.append(m_ap)
                        pred_scores = []
                        gt_labels = []
                    vid = clip_info["vid"]

                pred_scores.append(clip_info["pred_score"])
                gt_labels.append(clip_info["clip_label"])

            if len(gt_labels) > 0:
                auc, m_ap = self.calculate_metrics(gt_labels, pred_scores)
                test_aucs.append(auc)
                test_m_aps.append(m_ap)

            test_loss = float(np.mean(losses))
            test_auc = float(np.mean(test_aucs))
            test_m_ap = float(np.mean(test_m_aps))

            print(f"{split}, loss: {test_loss}, auc {test_auc}, m_ap {test_m_ap}")

            metrics_dict = {
                'loss': test_loss,
                'auc': test_auc,
                'map': test_m_ap
            }
            self.log_metrics(split, metrics_dict, epoch)
            
            return test_m_ap

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu_ids', default='0,1,2,3,4', type=str,help='gpu ids to use (e.g., "0,1,2"). -1 for CPU')
    parser.add_argument('--data_mode', default="all", type=str)
    parser.add_argument('--model_type', default="two_stream", type=str)
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--epoch', default=280, type=int)
    parser.add_argument('--batch_size', default=80, type=int)
    parser.add_argument('--window_size', default=1, type=int)
    args = parser.parse_args()

    # Device configuration
    if args.gpu_ids == '-1':
        device = torch.device('cpu')
        gpu_ids = []
    else:
        gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
        
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Using GPU ids: {gpu_ids}")

    set_random_seed.use_fix_random_seed()
    batch_size = args.batch_size
    clip_frame_num = args.clip_frame_num
    window_size = args.window_size
    num_workers = 4
    max_text_len = 100

    # Paths
    vision_pretrain_ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/r50tsm/batch_{batch_size}_lr_decay_cosine_train_test_split/pretrain.pth"
    lang_pretrain_ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/hugface_bert_pretrain/batch_{batch_size}_lr_decay_cosine_train_test_split/pretrain.pth"
    ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/chapter_localization/cross_window_attn_batch_{batch_size}_frame_{clip_frame_num}/checkpoint.pth"
    img_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/youtube_video_frame_dataset"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_final.csv"
    test_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/validation_clips_clip_frame_num_{clip_frame_num}.json"

    train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_train.txt"
    test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_validation.txt"
    # train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/debugging_train.txt"
    tensorboard_log = os.path.dirname(ckpt_path)
    tensorboard_writer = SummaryWriter(tensorboard_log)

    # init model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    lang_model = bert_hugface.BertHugface(pretrain_stage=False)
    vision_model = resnet50_tsm.Resnet50TSM(
        segments_size=args.clip_frame_num,
        shift_div=8,
        pretrain_stage=False
    )

    if os.path.exists(lang_pretrain_ckpt_path):
        lang_state_dict = torch.load(lang_pretrain_ckpt_path, map_location=device)
        lang_model.load_state_dict(lang_state_dict)
    else:
        print(f"Warning: Language model checkpoint not found at {lang_pretrain_ckpt_path}")
    gc.collect()

    if os.path.exists(vision_pretrain_ckpt_path):
        vision_state_dict = torch.load(vision_pretrain_ckpt_path, map_location=device)
        vision_model.load_state_dict(vision_state_dict)
    else:
        print(f"Warning: Vision model checkpoint not found at {vision_pretrain_ckpt_path}")
    gc.collect()


    # GlobalTwoStream model
    if args.data_mode == "all":
        hidden_size = 128
        lang_base_model = lang_model.base_model
        vision_base_model = vision_model.base_model

        model = two_stream_window_optim_sep.TwoStream(
            lang_model=lang_base_model,
            vision_model=vision_base_model,
            lang_embed_size=lang_model.embed_size,
            vision_embed_size=vision_model.feature_dim,
            segment_size=args.clip_frame_num,
            hidden_size=hidden_size,
            window_size = args.window_size
        )
        model.build_chapter_head(output_size=2)
        model = model.to(device)
        if torch.cuda.is_available() and len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    else:
        raise RuntimeError(f"Unknown data mode {args.data_mode}")

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

    train_dataset = WindowClipDataset(
        img_dir=img_dir,
        data_file=data_file,
        vid_file=train_vid_file,
        tokenizer=tokenizer,
        clip_frame_num=args.clip_frame_num,
        max_text_len=max_text_len,
        window_size=window_size,
        mode=args.data_mode,
        transform=train_vision_preprocess
    )

    test_dataset = InferWindowClipDataset(
        img_dir=img_dir,
        json_paths=test_clips_json,
        tokenizer=tokenizer,
        clip_frame_num=args.clip_frame_num,
        max_text_len=max_text_len,
        window_size=window_size,
        mode=args.data_mode,
        transform=test_vision_preprocess
    )

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(
        data_mode=args.data_mode,
        max_epochs=args.epoch,
        batch_size=args.batch_size,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        validation_interval=20,
        num_workers=num_workers,
        ckpt_path=ckpt_path,
        tensorboard_writer=tensorboard_writer
    )

    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.device = device
    trainer.train()