"""
Train models (language, vision or both) on youtube dataset
"""

import math
import os
import time
import logging
import glob
import re

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from data.youtube_dataset import YoutubeClipDataset, WindowClipDataset 
from data.infer_youtube_video_dataset import InferYoutubeClipDataset, InferWindowClipDataset
from model.lang import bert_hugface
from model.vision import resnet50_tsm, resnet50
from model.fusion import two_stream_window
# from model.fusion import two_stream_domain_specific
from common_utils import set_random_seed
from sklearn import metrics
from memory_cache_utils import MemoryManager


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
    gradient_accumulation_steps = 4

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
        self.memory_manager = MemoryManager()

        # take over whatever gpus are on the system
        # self.device = 'cpu'
        # if torch.cuda.is_available():
            # self.device = torch.cuda.current_device()
            # self.model = self.model.to(self.device)
            # self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch, best_result, is_best=True):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        # checkpoint_dir = os.path.dirname(self.config.ckpt_path)
        checkpoint_dir = self.config.ckpt_path
        os.makedirs(checkpoint_dir, exist_ok=True)

        if is_best:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch}_{best_result:.4f}.pth")
        else:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch}.pth")
        
        logger.info(f"Saving checkpoint at epoch {epoch} to {ckpt_path}")

        checkpoint = {
                "epoch": epoch,
                "best_result": best_result,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }

        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def find_latest_checkpoint(self):
        if not os.path.exists(self.config.ckpt_path):
            return None, 0
            
        checkpoints = glob.glob(os.path.join(self.config.ckpt_path, "*.pth"))
        if not checkpoints:
            return None, 0

        latest_epoch = -1
        latest_checkpoint = None
        
        for checkpoint in checkpoints:
            filename = os.path.basename(checkpoint)
            match = re.search(r'ckpt_epoch(\d+)(?:_[\d.]+)?\.pth$', filename)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = checkpoint
        
        return latest_checkpoint, latest_epoch

    def train(self):

        latest_checkpoint, start_epoch = self.find_latest_checkpoint()
    
        if latest_checkpoint:
            checkpoint = torch.load(latest_checkpoint)
            state_dict = checkpoint["model_state_dict"]

            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('module.'):
                    k = 'module.' + k 
                new_state_dict[k] = v
                
            self.config.start_epoch = checkpoint["epoch"]
            self.config.best_result = checkpoint["best_result"]
            self.model.load_state_dict(new_state_dict)
            logger.info(f"Resuming from epoch {self.config.start_epoch} with best result {self.config.best_result}")
        else:
            self.config.start_epoch = 0
            self.config.best_result = float('-inf')
            logger.info("No checkpoint found, starting from scratch")

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.config)
        if latest_checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        best_result = self.config.best_result
        test_result = float('-inf')
        self.tokens = 0

        for epoch in range(self.config.start_epoch+1, self.config.max_epochs+1):
            self.run_epoch('train', epoch, self.train_dataset)

            if self.test_dataset is not None and ((epoch % 30 == 0) or (epoch == 15) or (epoch == 45)):
                infer_test_result = self.run_epoch("infer_test", epoch, self.test_dataset)
                test_result = infer_test_result

                if test_result > best_result:
                    best_result = test_result
                    if self.config.ckpt_path is not None:
                        checkpoint_path = self.save_checkpoint(epoch, best_result, is_best=True)
                        logger.info(f"Saved best model checkpoint to {checkpoint_path}")

            # elif epoch % 10 == 0 and self.config.ckpt_path is not None:
            #     checkpoint_path = self.save_checkpoint(epoch, best_result, is_best=False)
            #     logger.info(f"Saved regular checkpoint to {checkpoint_path}")


    def run_epoch(self, split, epoch, dataset):
        is_train = split == 'train'
        self.model.train(is_train)

        if is_train:
            shuffle = True
            used_batch_size = self.config.batch_size
        else:
            shuffle = False
            used_batch_size = self.config.batch_size * 8

        losses = []
        
        loader = DataLoader(dataset, shuffle=shuffle, pin_memory=True, batch_size=used_batch_size, num_workers=self.config.num_workers)
        # pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        pbar = tqdm(enumerate(loader), total=len(loader))

        # lr = self.config.learning_rate

        # for it, (img_clip, text_ids, attention_mask, label) in pbar:
        for it, (img_clip, text_ids, attention_mask, label, clip_info) in pbar:
            img_clip = img_clip.float().to(self.device)
            text_ids = text_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)   
            label = label.to(self.device)
            clip_info = {k: v.to(self.device) for k, v in clip_info.items()}

            # forward the model
            with torch.set_grad_enabled(is_train):
                if self.config.data_mode == "text":
                    binary_logits, binary_prob = self.model(text_ids, attention_mask)
                elif self.config.data_mode == "image":
                    binary_logits, binary_prob = self.model(img_clip)
                elif self.config.data_mode == "all":
                    # binary_logits, binary_prob = self.model(img_clip, text_ids, attention_mask)    
                    binary_logits, binary_prob = self.model(img_clip, text_ids, attention_mask, clip_info)    
                else:
                    raise RuntimeError(f"Unknown data mode {self.config.data_mode}")
                
                loss = F.cross_entropy(binary_logits, label)

            if not is_train:
                scores = binary_prob[:, 1].detach().cpu().numpy()
                start_idx = it * used_batch_size
                end_idx = start_idx + img_clip.shape[0]

                for i in range(start_idx, end_idx):
                    pred_idx = i - start_idx
                    dataset.all_clip_infos[i]["pred_score"] = scores[pred_idx]

                losses.append(loss.item())
                
            if is_train:
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # gradient accumulation
                if (it + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                    self.optimizer.step()
                    self.model.zero_grad()

                    # decay the learning rate based on our progress
                    if self.config.lr_decay:

                        if epoch < self.config.warmup_epochs:
                            lr_mult = max(epoch / self.config.warmup_epochs, 1e-2)
                        else:
                            if epoch < self.config.final_epochs:
                                progress = epoch / self.config.final_epochs
                            else:
                                progress = 1.0

                            if self.config.lr_decay_type == "cosine":
                                lr_mult = max(0.001, 0.5 * (1.0 + math.cos(math.pi * progress)))
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
                        # for param_group in self.optimizer.param_groups:
                        #     param_group['lr'] = param_group['initial_lr'] * lr_mult
                    else:
                        lr = self.config.learning_rate
                        # for param_group in self.optimizer.param_groups:
                        #     if 'initial_lr' in param_group:
                        #         param_group['lr'] = param_group['initial_lr']
                        #     else:
                        #         param_group['lr'] = self.config.learning_rate

                    cpu_y = list(label.cpu().numpy())
                    scores = binary_prob[:, 1].detach().cpu().numpy()
                    pos_num = cpu_y.count(1)

                    if len(cpu_y) > pos_num > 0:
                        fpr, tpr, thresholds = metrics.roc_curve(cpu_y, scores, pos_label=1)
                        auc = metrics.auc(fpr, tpr)
                        m_ap = metrics.average_precision_score(cpu_y, scores)
                    else:
                        auc = 0
                        m_ap = 0

                    # report progress
                    n_iter = epoch * len(loader) + it
                    if self.config.tensorboard_writer:
                        self.config.tensorboard_writer.add_scalar('Train/loss', loss.item(), n_iter)
                        if auc:
                            self.config.tensorboard_writer.add_scalar('Train/auc', auc, n_iter)
                            self.config.tensorboard_writer.add_scalar('Train/m_ap', m_ap, n_iter)

                    pbar.set_description(
                        f"epoch {epoch} iter {it}: train loss {loss.item():.5f}, "
                        f"auc {auc:.5f}, m_ap {m_ap:.5f}"
                    )

                # else:
                #     pbar.set_description(
                #         f"epoch {epoch} iter {it}: train loss {loss.item():.5f}, "
                #         f"grad_accum_step {(it + 1) % self.config.gradient_accumulation_steps}"
                #     )

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

            self.memory_manager.cleanup(force=True)

            return test_m_ap



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_mode', default="all", type=str, help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', default="two_stream", type=str, help="bert, r50tsm, two_stream")
    parser.add_argument('--clip_frame_num', default=8, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--head_type', default="mlp", type=str, help="mlp, bilinear, self_attn, cross_attn, only work on two_stream model")
    parser.add_argument('--window_size', default=2, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    args = parser.parse_args()

    set_random_seed.use_fix_random_seed()
    batch_size = args.batch_size
    gradient_accumulation_steps=args.gradient_accumulation_steps
    clip_frame_num = args.clip_frame_num
    num_workers = 16
    max_text_len = 100
    start_epoch = 0
    best_result = float('-inf')

    vision_pretrain_ckpt_path = f"./checkpoint/r50tsm/batch_{batch_size}_lr_decay_cosine_train_test_split/pretrain.pth"
    # lang_pretrain_ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/hugface_bert_pretrain/batch_{batch_size}_lr_decay_cosine_train_test_split/pretrain.pth"
    lang_pretrain_ckpt_path = f"./checkpoint/hugface_bert/pretrain_2880.pth"
    # ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/{args.data_mode}/{args.model_type}_validation/batch_{batch_size}_head_type_{args.head_type}_clip_frame_num_{args.clip_frame_num}/checkpoint.pth"
    ckpt_path = f"./checkpoint/chapter_localization/window_attnX6/clip{clip_frame_num}_w{args.window_size}"
    img_dir = "./youtube_video_frame_dataset"
    data_file = "./dataset/all_in_one_with_subtitle_final.csv"
    subtitle_dir = "../video_chapter_youtube_dataset/dataset"
    test_clips_json = f"./dataset/validation_clips_clip_frame_num_{clip_frame_num}.json"

    train_vid_file = "./dataset/final_train.txt"
    test_vid_file = "./dataset/final_validation.txt"
    # tensorboard_log = os.path.dirname(ckpt_path)
    tensorboard_log = ckpt_path
    tensorboard_writer = SummaryWriter(tensorboard_log)

    # init model
    # lang model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    lang_model = bert_hugface.BertHugface(pretrain_stage=False)
    if os.path.exists(lang_pretrain_ckpt_path):
        # lang_model.load_state_dict(torch.load(lang_pretrain_ckpt_path))        
        checkpoint = torch.load(lang_pretrain_ckpt_path)
        lang_model.load_state_dict(checkpoint["model_state_dict"])

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
        # model = two_stream_update.TwoStream(
        #     lang_base_model,
        #     vision_base_model,
        #     lang_model.embed_size,
        #     vision_model.feature_dim,
        #     clip_frame_num,
        #     hidden_size)
        model = two_stream_window.TwoStream(
            lang_base_model,
            vision_base_model,
            lang_model.embed_size,
            vision_model.feature_dim,
            clip_frame_num,
            hidden_size,
            args.window_size)
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
    
    # train_dataset = YoutubeClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=train_vision_preprocess)
    # test_dataset = InferYoutubeClipDataset(img_dir, test_clips_json, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)
    train_dataset = WindowClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, window_size=args.window_size, mode=args.data_mode, transform=train_vision_preprocess, subtitle_dir=subtitle_dir)
    test_dataset = InferWindowClipDataset(img_dir, test_clips_json, tokenizer, clip_frame_num, max_text_len, window_size=args.window_size, mode=args.data_mode, transform=test_vision_preprocess)

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(data_mode=args.data_mode, max_epochs=args.epoch,
                        start_epoch=start_epoch, best_result=best_result, batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=1e-5, block_size=max_text_len,
                        lr_decay_type=args.lr_decay_type, lr_decay=True, warmup_epochs=args.epoch//100, final_epochs=args.epoch//100*90, 
                        num_workers=num_workers, ckpt_path=ckpt_path, tensorboard_writer=tensorboard_writer)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.device = args.gpu
    trainer.train()
