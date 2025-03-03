"""
Train models (language, vision or both) on youtube dataset with DeepSpeed ZeRO Stage 2
"""

import math
import os
import time
import logging
import json

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from data.youtube_dataset import YoutubeClipDataset, YoutubeAllClipDataset
from data.infer_youtube_video_dataset import InferYoutubeClipDataset, InferYoutubeAllClipDataset
from model.lang import bert_hugface
from model.vision import resnet50_tsm, resnet50
from model.fusion import two_stream
from common_utils import set_random_seed

# Import deepspeed
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

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
        self.device = None  # Will be set later
        
        # Define DeepSpeed config if using DeepSpeed
        if self.config.use_deepspeed:
            self.setup_deepspeed_config()
            

    def setup_deepspeed_config(self):
        """Create DeepSpeed config if not provided externally"""
        if self.config.deepspeed_config is None:
            ds_config = {
                "train_batch_size": self.config.batch_size,
                "train_micro_batch_size_per_gpu": self.config.batch_size,
                "steps_per_print": 10000,
                
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e7, # 2e8
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
            ds_config_path = "ds_config.json"
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
        
        # For DeepSpeed checkpoints
        # base_path = os.path.splitext(self.config.ckpt_path)[0]
        if is_best:
            ds_ckpt_dir = f"{checkpoint_dir}/ds_checkpoint_{epoch}_score_{best_result:.4f}"
        else:
            ds_ckpt_dir = f"{checkpoint_dir}/ds_checkpoint_{epoch}"
            
        # Save using DeepSpeed's checkpoint mechanism
        self.model_engine.save_checkpoint(ds_ckpt_dir)
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
        else:
            # Standard PyTorch training
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            self.optimizer = raw_model.configure_optimizers(self.config)
            
        best_result = self.config.best_result
        test_result = float('-inf')
        
        for epoch in range(self.config.start_epoch+1, self.config.max_epochs+1):
            self.run_epoch('train', epoch, self.train_dataset)

            if self.test_dataset is not None and epoch % 30 == 0:
                infer_test_result = self.run_epoch("infer_test", epoch, self.test_dataset)
                test_result = infer_test_result

                if test_result > best_result:
                    best_result = test_result
                    if self.config.ckpt_path is not None:
                        client_state = {
                            'epoch': epoch,
                            'best_result': best_result
                        }
                        # When using DeepSpeed, pass client_state
                        if self.config.use_deepspeed:
                            self.model_engine.save_checkpoint(
                                save_dir=os.path.join(os.path.dirname(self.config.ckpt_path), f"ds_checkpoint_{epoch}_score_{best_result:.4f}"),
                                client_state=client_state
                            )
                            logger.info(f"Saved best model checkpoint at epoch {epoch} with score {best_result:.4f}")
                        else:
                            checkpoint_path = self.save_checkpoint(epoch, best_result, is_best=True)
                            logger.info(f"Saved best model checkpoint to {checkpoint_path}")

    def run_epoch(self, split, epoch, dataset):
        is_train = split == 'train'
        if is_train:
            if self.config.use_deepspeed:
                self.model_engine.train()
            else:
                self.model.train()
            shuffle = True
        else:
            if self.config.use_deepspeed:
                self.model_engine.eval()
            else:
                self.model.eval()
            shuffle = False

        losses = []
        
        loader = DataLoader(dataset, shuffle=shuffle, pin_memory=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
        pbar = tqdm(enumerate(loader), total=len(loader))

        for it, (img_clip, text_ids, attention_mask, label) in pbar:
            # Move tensors to device
            img_clip = img_clip.float().to(self.device)
            text_ids = text_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)   
            label = label.to(self.device)

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
                        binary_logits, binary_prob = self.model_engine(img_clip, text_ids, attention_mask)
                    else:
                        binary_logits, binary_prob = self.model(img_clip, text_ids, attention_mask)
                else:
                    raise RuntimeError(f"Unknown data mode {self.config.data_mode}")
                
                loss = F.cross_entropy(binary_logits, label)

            # Record results
            cpu_y = list(label.cpu().numpy())
            scores = binary_prob[:, 1].detach().cpu().numpy()

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

                for i in range(start_idx, min(end_idx, len(dataset.all_clip_infos))):
                    pred_idx = i - start_idx
                    dataset.all_clip_infos[i]["pred_score"] = scores[pred_idx]

                losses.append(loss.item())
                
            if is_train:
                # Backward pass and update parameters
                if self.config.use_deepspeed:
                    # DeepSpeed handles loss scaling and gradient clipping
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                else:
                    # Standard PyTorch training
                    self.model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                    self.optimizer.step()

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
                    if self.config.use_deepspeed:
                        for param_group in self.model_engine.optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                else:
                    lr = self.config.learning_rate

                # Report progress
                n_iter = epoch * len(loader) + it
                if self.config.tensorboard_writer:
                    self.config.tensorboard_writer.add_scalar('Train/loss', loss.item(), n_iter)
                    if auc:
                        self.config.tensorboard_writer.add_scalar('Train/auc', auc, n_iter)
                        self.config.tensorboard_writer.add_scalar('Train/m_ap', m_ap, n_iter)
                pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}, auc {auc:.5f}, m_ap {m_ap:.5f}, lr {lr:e}")

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

            # Process last video if needed
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
            
            if self.config.tensorboard_writer:
                self.config.tensorboard_writer.add_scalar(f'{split}/loss', test_loss, epoch)
                self.config.tensorboard_writer.add_scalar(f'{split}/auc', test_auc, epoch)
                self.config.tensorboard_writer.add_scalar(f'{split}/m_ap', test_m_ap, epoch)
                
            return test_m_ap


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model with DeepSpeed')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_mode', default="all", type=str, help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', default="two_stream", type=str, help="bert, r50tsm, two_stream")
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--epoch', default=280, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--head_type', default="mlp", type=str, help="mlp or attn, only work on two_stream model")
    parser.add_argument('--use_deepspeed', action='store_true', help="Enable DeepSpeed for training")
    parser.add_argument('--deepspeed_config', type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    set_random_seed.use_fix_random_seed()
    batch_size = args.batch_size
    clip_frame_num = args.clip_frame_num
    num_workers = 8
    max_text_len = 100
    start_epoch = 0
    best_result = float('-inf')

    vision_pretrain_ckpt_path = f"./checkpoint/r50tsm/batch_{batch_size}_lr_decay_cosine_train_test_split/pretrain.pth"
    lang_pretrain_ckpt_path = f"./checkpoint/hugface_bert/pretrain_2880.pth"
    ckpt_path = f"./checkpoint/chapter_localization/MVCG_deepspeed/test/"
    img_dir = "./dataset/youtube_video_frame_dataset"
    data_file = "./dataset/all_in_one_with_subtitle_final.csv"
    subtitle_dir = "../video_chapter_youtube_dataset/dataset"
    test_clips_json = f"./dataset/dataset_fps1/validation_clips_clip_frame_num_{clip_frame_num}.json"

    train_vid_file = "./dataset/final_train.txt"
    test_vid_file = "./dataset/final_validation.txt"
    tensorboard_log = os.path.dirname(ckpt_path)
    os.makedirs(tensorboard_log, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_log)

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
        model = two_stream.TwoStream(lang_base_model, vision_base_model, lang_model.embed_size, vision_model.feature_dim, clip_frame_num, hidden_size)
        model.build_chapter_head(output_size=2, head_type=args.head_type)
        model = model.to(args.gpu)
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
    
    train_dataset = YoutubeClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=train_vision_preprocess, subtitle_dir=subtitle_dir)
    test_dataset = InferYoutubeClipDataset(img_dir, test_clips_json, tokenizer, clip_frame_num, max_text_len, mode=args.data_mode, transform=test_vision_preprocess)

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(
        data_mode=args.data_mode, 
        max_epochs=args.epoch,
        start_epoch=start_epoch, 
        best_result=best_result, 
        batch_size=batch_size, 
        learning_rate=1e-5, 
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