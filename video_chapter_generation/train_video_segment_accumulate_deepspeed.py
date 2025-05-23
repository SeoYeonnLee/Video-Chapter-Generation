"""
Train models (language, vision or both) on youtube dataset with DeepSpeed
"""

import math
import os
import time
import logging
import glob
import re
import json

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
from common_utils import set_random_seed
from sklearn import metrics
from memory_cache_utils import MemoryManager

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

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
    gradient_accumulation_steps = 2
    
    # DeepSpeed config
    use_deepspeed = True
    deepspeed_config = None

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
        self.device = None  # Will be set later

        # Define DeepSpeed config if using DeepSpeed
        if self.config.use_deepspeed:
            self.setup_deepspeed_config()

    def setup_deepspeed_config(self):
        """Create DeepSpeed config if not provided externally"""
        if self.config.deepspeed_config is None:
            ds_config = {
                "train_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps,
                "train_micro_batch_size_per_gpu": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "steps_per_print": 10000,
                
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e7,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e7,
                    "contiguous_gradients": True,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    }
                },
                
                "gradient_clipping": self.config.grad_norm_clip,
                
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": self.config.learning_rate,
                        "betas": self.config.betas,
                        "eps": 1e-8,
                        "weight_decay": self.config.weight_decay
                    }
                },
                
                "fp16": {
                    "enabled": False
                },
                
                "zero_allow_untested_optimizer": True,

                "verbose": False,             
                "wall_clock_breakdown": False,
                "timer": {
                    "enabled": False        
                }
            }
            
            # Save the config to a file for DeepSpeed to use
            ds_config_path = "ds_config_window.json"
            with open(ds_config_path, 'w') as f:
                json.dump(ds_config, f, indent=4)
            
            self.config.deepspeed_config = ds_config_path

    def save_checkpoint(self, epoch, best_result, is_best=True):
        """Save model checkpoint with DeepSpeed"""
        if not hasattr(self, 'model_engine'):
            logger.error("DeepSpeed model engine not initialized, cannot save checkpoint")
            return
            
        checkpoint_dir = os.path.dirname(self.config.ckpt_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            ds_ckpt_dir = f"{checkpoint_dir}/ds_checkpoint_{epoch}_score_{best_result:.4f}"
        else:
            ds_ckpt_dir = f"{checkpoint_dir}/ds_checkpoint_{epoch}"
        
        # Create client state to include epoch and best_result
        client_state = {
            "epoch": epoch,
            "best_result": best_result
        }
        
        # Save using DeepSpeed's checkpoint mechanism
        self.model_engine.save_checkpoint(ds_ckpt_dir, client_state=client_state)
        logger.info(f"Saved DeepSpeed checkpoint at epoch {epoch} to {ds_ckpt_dir}")
        
        return ds_ckpt_dir

    def load_checkpoint(self, checkpoint_dir):
        """Load model from a DeepSpeed checkpoint directory"""
        if not os.path.isdir(checkpoint_dir):
            logger.error(f"Checkpoint directory {checkpoint_dir} does not exist")
            return False
            
        # DeepSpeed will load the model and optimizer states
        _, client_state = self.model_engine.load_checkpoint(checkpoint_dir)
        
        # Get epoch and best result from client state
        if client_state:
            self.config.start_epoch = client_state.get('epoch', 0)
            self.config.best_result = client_state.get('best_result', float('-inf'))
            logger.info(f"Loaded checkpoint: epoch {self.config.start_epoch}, best result {self.config.best_result:.4f}")
            return True
        
        return False

    def find_latest_deepspeed_checkpoint(self):
        """Find the latest DeepSpeed checkpoint in the checkpoint directory"""
        checkpoint_dir = os.path.dirname(self.config.ckpt_path)
        if not os.path.exists(checkpoint_dir):
            return None
            
        checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "ds_checkpoint_*"))
        if not checkpoint_dirs:
            return None

        latest_epoch = -1
        latest_checkpoint = None
        
        for ckpt_dir in checkpoint_dirs:
            dir_name = os.path.basename(ckpt_dir)
            match = re.search(r'ds_checkpoint_(\d+)(?:_score_[\d.]+)?$', dir_name)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = ckpt_dir
        
        return latest_checkpoint

    def train(self):
        # Initialize DeepSpeed if enabled
        if self.config.use_deepspeed:
            # Initialize DeepSpeed
            parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            
            model_engine, optimizer, _, _ = deepspeed.initialize(
                args=argparse.Namespace(**{"local_rank": 0}),  # For single GPU
                model=self.model,
                model_parameters=parameters,
                config=self.config.deepspeed_config
            )
            
            self.model_engine = model_engine
            self.optimizer = optimizer
            self.device = model_engine.device
            
            logger.info(f"DeepSpeed initialized with device {self.device}")
            
            # Try to load the latest checkpoint
            latest_checkpoint = self.find_latest_deepspeed_checkpoint()
            if latest_checkpoint:
                success = self.load_checkpoint(latest_checkpoint)
                if success:
                    logger.info(f"Resuming from DeepSpeed checkpoint: {latest_checkpoint}")
                else:
                    logger.warning(f"Failed to load DeepSpeed checkpoint: {latest_checkpoint}")
                    self.config.start_epoch = 0
                    self.config.best_result = float('-inf')
            else:
                logger.info("No DeepSpeed checkpoint found, starting from scratch")
                self.config.start_epoch = 0
                self.config.best_result = float('-inf')
        else:
            # Standard PyTorch training with checkpoint loading
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            self.optimizer = raw_model.configure_optimizers(self.config)
            
            # Load the latest checkpoint if available
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
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info(f"Resuming from epoch {self.config.start_epoch} with best result {self.config.best_result}")
            else:
                self.config.start_epoch = 0
                self.config.best_result = float('-inf')
                logger.info("No checkpoint found, starting from scratch")

        best_result = self.config.best_result
        test_result = float('-inf')
        
        for epoch in range(self.config.start_epoch+1, self.config.max_epochs+1):
            self.run_epoch('train', epoch, self.train_dataset)

            if self.test_dataset is not None and (epoch % 30 == 0):
                infer_test_result = self.run_epoch("infer_test", epoch, self.test_dataset)
                test_result = infer_test_result

                if test_result > best_result:
                    best_result = test_result
                    if self.config.ckpt_path is not None:
                        if self.config.use_deepspeed:
                            checkpoint_path = self.save_checkpoint(epoch, best_result, is_best=True)
                        else:
                            checkpoint_path = self.save_checkpoint(epoch, best_result, is_best=True)
                        logger.info(f"Saved best model checkpoint to {checkpoint_path}")

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
            match = re.search(r'_(\d+)(?:_score_[\d.]+)?\.pth$', filename)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = checkpoint
        
        return latest_checkpoint, latest_epoch

    def run_epoch(self, split, epoch, dataset):
        is_train = split == 'train'
        
        if is_train:
            if self.config.use_deepspeed:
                self.model_engine.train()
            else:
                self.model.train()
            shuffle = True
            used_batch_size = self.config.batch_size
        else:
            if self.config.use_deepspeed:
                self.model_engine.eval()
            else:
                self.model.eval()
            shuffle = False
            used_batch_size = self.config.batch_size * 8

        losses = []
        
        loader = DataLoader(dataset, shuffle=shuffle, pin_memory=True, batch_size=used_batch_size, num_workers=self.config.num_workers)
        pbar = tqdm(enumerate(loader), total=len(loader))

        for it, (img_clip, text_ids, attention_mask, label, clip_info) in pbar:
            # Move tensors to device
            img_clip = img_clip.float().to(self.device)
            text_ids = text_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)   
            label = label.to(self.device)
            clip_info = {k: v.to(self.device) for k, v in clip_info.items()}

            # Forward the model
            with torch.set_grad_enabled(is_train):
                if self.config.data_mode == "text":
                    if self.config.use_deepspeed:
                        binary_logits, binary_prob = self.model_engine(text_ids, attention_mask)
                    else:
                        binary_logits, binary_prob = self.model(text_ids, attention_mask)
                        
                elif self.config.data_mode == "image":
                    if self.config.use_deepspeed:
                        binary_logits, binary_prob = self.model_engine(img_clip)
                    else:
                        binary_logits, binary_prob = self.model(img_clip)
                        
                elif self.config.data_mode == "all":
                    if self.config.use_deepspeed:
                        binary_logits, binary_prob = self.model_engine(img_clip, text_ids, attention_mask, clip_info)
                    else:
                        binary_logits, binary_prob = self.model(img_clip, text_ids, attention_mask, clip_info)
                else:
                    raise RuntimeError(f"Unknown data mode {self.config.data_mode}")
                
                loss = F.cross_entropy(binary_logits, label)

            # Record results
            if not is_train:
                scores = binary_prob[:, 1].detach().cpu().numpy()
                start_idx = it * used_batch_size
                end_idx = start_idx + img_clip.shape[0]

                for i in range(start_idx, min(end_idx, len(dataset.all_clip_infos))):
                    pred_idx = i - start_idx
                    dataset.all_clip_infos[i]["pred_score"] = scores[pred_idx]

                losses.append(loss.item())
                
            if is_train:
                if self.config.use_deepspeed:
                    # DeepSpeed handles loss scaling and backward
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                    
                    # Get metrics for logging
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
                    
                    # Get current learning rate
                    if hasattr(self.model_engine.optimizer, 'param_groups'):
                        lr = self.model_engine.optimizer.param_groups[0]['lr']
                    else:
                        lr = self.config.learning_rate
                    
                    # Report progress
                    n_iter = epoch * len(loader) + it
                    if self.config.tensorboard_writer:
                        self.config.tensorboard_writer.add_scalar('Train/loss', loss.item(), n_iter)
                        if auc:
                            self.config.tensorboard_writer.add_scalar('Train/auc', auc, n_iter)
                            self.config.tensorboard_writer.add_scalar('Train/m_ap', m_ap, n_iter)

                    pbar.set_description(
                        f"epoch {epoch} iter {it}: train loss {loss.item():.5f}, "
                        f"auc {auc:.5f}, m_ap {m_ap:.5f}, lr {lr:e}"
                    )
                else:
                    # Standard PyTorch training with gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()

                    # Gradient accumulation
                    if (it + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                        self.optimizer.step()
                        self.model.zero_grad()

                        # Learning rate scheduling
                        if self.config.lr_decay:
                            if epoch < self.config.warmup_epochs:
                                # Linear warmup
                                lr_mult = max(epoch / self.config.warmup_epochs, 1e-2)
                            else:
                                if epoch < self.config.final_epochs:
                                    progress = epoch / self.config.final_epochs
                                else:
                                    progress = 1.0
                                # Cosine learning rate decay
                                if self.config.lr_decay_type == "cosine":
                                    lr_mult = max(0.001, 0.5 * (1.0 + math.cos(math.pi * progress)))
                                # Exponential learning rate decay
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

                        # Calculate metrics
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

                        # Report progress
                        n_iter = epoch * len(loader) + it
                        if self.config.tensorboard_writer:
                            self.config.tensorboard_writer.add_scalar('Train/loss', loss.item(), n_iter)
                            if auc:
                                self.config.tensorboard_writer.add_scalar('Train/auc', auc, n_iter)
                                self.config.tensorboard_writer.add_scalar('Train/m_ap', m_ap, n_iter)

                        pbar.set_description(
                            f"epoch {epoch} iter {it}: train loss {loss.item():.5f}, "
                            f"auc {auc:.5f}, m_ap {m_ap:.5f}, lr {lr:e}"
                        )

        # Process test results
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

                pred_scores.append(clip_info["pred_score"])
                gt_labels.append(clip_info["clip_label"])

            # Process last video
            if len(gt_labels) > 0:
                fpr, tpr, thresholds = metrics.roc_curve(gt_labels, pred_scores, pos_label=1)
                test_auc = metrics.auc(fpr, tpr)
                test_m_ap = metrics.average_precision_score(gt_labels, pred_scores)
                test_aucs.append(test_auc)
                test_m_aps.append(test_m_ap)

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
    parser = argparse.ArgumentParser(description='video chapter model with DeepSpeed')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_mode', default="all", type=str, help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', default="two_stream", type=str, help="bert, r50tsm, two_stream")
    parser.add_argument('--clip_frame_num', default=12, type=int)
    parser.add_argument('--epoch', default=270, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--head_type', default="cross_attn", type=str, help="mlp, self_attn, cross_attn, only work on two_stream model")
    parser.add_argument('--window_size', default=1, type=int)
    parser.add_argument('--use_deepspeed', action='store_true', help="Enable DeepSpeed for training")
    parser.add_argument('--deepspeed_config', type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank for distributed training")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps")
    args = parser.parse_args()

    set_random_seed.use_fix_random_seed()
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    clip_frame_num = args.clip_frame_num
    num_workers = 4
    max_text_len = 100
    start_epoch = 0
    best_result = float('-inf')

    vision_pretrain_ckpt_path = f"./checkpoint/r50tsm/batch_{batch_size}_lr_decay_cosine_train_test_split/pretrain.pth"
    lang_pretrain_ckpt_path = f"./checkpoint/hugface_bert/pretrain_2880.pth"
    ckpt_path = f"./checkpoint/chapter_localization/MVCG_deepspeed_lr_2e-6_fullval/test/"
    img_dir = "./dataset/youtube_video_frame_dataset"
    data_file = "./dataset/all_in_one_with_subtitle_final.csv"
    subtitle_dir = "../video_chapter_youtube_dataset/dataset"
    test_clips_json = f"./dataset/dataset_fps1/validation_clips_clip_frame_num_12.json"

    train_vid_file = "./dataset/final_train.txt"
    test_vid_file = "./dataset/final_validation.txt"
    tensorboard_log = os.path.dirname(ckpt_path)
    os.makedirs(tensorboard_log, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_log)

    try:
        # init model
        # lang model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        lang_model = bert_hugface.BertHugface(pretrain_stage=False)
        if os.path.exists(lang_pretrain_ckpt_path):
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
        elif args.data_mode == "all":
            lang_base_model = lang_model.base_model
            vision_base_model = vision_model.base_model
            hidden_size = 128
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
            
            # Only wrap in DataParallel if not using DeepSpeed
            if not args.use_deepspeed:
                model = torch.nn.DataParallel(model, device_ids=[0,1])
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
        
        train_dataset = WindowClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, window_size=args.window_size, mode=args.data_mode, transform=train_vision_preprocess, subtitle_dir=subtitle_dir)
        test_dataset = InferWindowClipDataset(img_dir, test_clips_json, tokenizer, clip_frame_num, max_text_len, window_size=args.window_size, mode=args.data_mode, transform=test_vision_preprocess)

        # initialize a trainer instance and kick off training
        tconf = TrainerConfig(
            data_mode=args.data_mode, 
            max_epochs=args.epoch,
            start_epoch=start_epoch, 
            best_result=best_result, 
            batch_size=batch_size, 
            gradient_accumulation_steps=gradient_accumulation_steps, 
            learning_rate=2e-6, 
            block_size=max_text_len,
            lr_decay_type=args.lr_decay_type, 
            lr_decay=True, 
            warmup_epochs=args.epoch//100, 
            final_epochs=args.epoch//100*90, 
            num_workers=num_workers, 
            ckpt_path=ckpt_path, 
            tensorboard_writer=tensorboard_writer,
            use_deepspeed=args.use_deepspeed,
            deepspeed_config=args.deepspeed_config
        )
        
        trainer = Trainer(model, train_dataset, test_dataset, tconf)
        trainer.device = f'cuda:{args.gpu}'
        trainer.train()

    finally:
        trainer.memory_manager.shutdown()