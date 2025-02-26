"""
Train models (language, vision or both) on youtube dataset
"""
import argparse
import platform
import socket
import math
import os
import time
import logging
import glob
import re
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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

start_time = time.time()

logger = logging.getLogger(__name__)

def setup_ddp(rank, world_size):
    """
    Setup DDP environment for single machine with multiple GPUs
    Auto-configure IP, port, and backend
    """
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def get_backend():
        if torch.cuda.is_available():
            return 'nccl'
        return 'gloo'

    # 자동 설정
    port = find_free_port()
    ip = get_ip()
    backend = get_backend()
    
    print(f"DDP Configuration - IP: {ip}, Port: {port}, Backend: {backend}, Rank: {rank}")
    
    # 환경 변수 설정
    os.environ['MASTER_ADDR'] = ip
    os.environ['MASTER_PORT'] = str(port)
    
    # 프로세스 그룹 초기화
    dist.init_process_group(
        backend=backend,
        init_method=f'tcp://{ip}:{port}',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    
def cleanup_ddp():
    """
    Clean up DDP environment
    """
    dist.destroy_process_group()


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

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, tokenizer, train_dataset, test_dataset, config, rank, world_size):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.memory_manager = MemoryManager()

        setup_ddp(rank, world_size)

        self.device = torch.device(f'cuda:{rank}')
        self.model = self.model.to(self.device)

        dist.barrier()
        self.model = DDP(self.model, device_ids=[rank])


    def save_checkpoint(self, epoch, best_result, is_best=False):
        if self.rank != 0:
            return None
        
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint_dir = os.path.dirname(self.config.ckpt_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        base_path = os.path.splitext(self.config.ckpt_path)[0]

        if is_best:
            ckpt_path = f"{base_path}_{epoch}_score_{best_result:.4f}.pth"
        else:
            ckpt_path = f"{base_path}_{epoch}.pth"
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
        # rank 0에서만 체크포인트를 찾고 결과를 다른 rank와 동기화
        if self.rank == 0:
            if not os.path.exists(self.config.ckpt_path):
                checkpoint_info = (None, 0)
            else:
                checkpoints = glob.glob(os.path.join(self.config.ckpt_path, "*.pth"))
                if not checkpoints:
                    checkpoint_info = (None, 0)
                else:
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
                    checkpoint_info = (latest_checkpoint, latest_epoch)
        else:
            checkpoint_info = (None, 0)

        # 결과 동기화
        if torch.distributed.is_initialized():
            checkpoint_info = [checkpoint_info]
            torch.distributed.broadcast_object_list(checkpoint_info, src=0)
            checkpoint_info = checkpoint_info[0]
            
        return checkpoint_info

    def train(self):
        # Create DistributedSampler for the datasets
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        if self.test_dataset is not None:
            test_sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )

        # Modify DataLoader to use DistributedSampler
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        if self.test_dataset is not None:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config.val_batch_size,
                sampler=test_sampler,
                num_workers=self.config.num_workers,
                pin_memory=True
            )

        latest_checkpoint, start_epoch = self.find_latest_checkpoint()
        if latest_checkpoint and self.rank ==0:
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
            self.config.start_epoch = checkpoint["epoch"]
            self.config.best_result = checkpoint["best_result"]
            logger.info(f"Resuming from epoch {self.config.start_epoch} with best result {self.config.best_result}")
        else:
            self.config.start_epoch = 0
            self.config.best_result = float('-inf')
            logger.info("No checkpoint found, starting from scratch")

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.config)
        if latest_checkpoint and self.rank == 0:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        dist.barrier()
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
            
        best_result = self.config.best_result
        test_result = float('-inf')
        self.tokens = 0

        for epoch in range(self.config.start_epoch+1, self.config.max_epochs+1):
            train_sampler.set_epoch(epoch)
            self.run_epoch('train', epoch, train_loader)

            if self.test_dataset is not None and (epoch % 30 == 0):
                dist.barrier()
                infer_test_result = self.run_epoch("infer_test", epoch, test_loader)

                gathered_results = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered_results, infer_test_result)

                if self.rank == 0:
                    test_result = sum(gathered_results) / len(gathered_results)
                    if test_result > best_result:
                        best_result = test_result
                        if self.config.ckpt_path is not None:
                            checkpoint_path = self.save_checkpoint(epoch, best_result, is_best=True)
                            logger.info(f"Saved best model checkpoint to {checkpoint_path}")

            elif epoch % 10 == 0 and self.config.ckpt_path is not None:
                checkpoint_path = self.save_checkpoint(epoch, best_result, is_best=False)
                logger.info(f"Saved regular checkpoint to {checkpoint_path}")
        
        cleanup_ddp()

    def run_epoch(self, split, epoch, loader):
        is_train = split == 'train'
        self.model.train(is_train)
        losses = []

        pbar = tqdm(enumerate(loader), total=len(loader)) if self.rank == 0 else enumerate(loader)
        
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
                start_idx = it * loader.batch_size
                end_idx = start_idx + img_clip.shape[0]

                for i in range(start_idx, end_idx):
                    pred_idx = i - start_idx
                    loader.dataset.all_clip_infos[i]["pred_score"] = scores[pred_idx]

                losses.append(loss.item())
                
            if is_train:
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                # gradient accumulation
                if (it + 1) % self.config.gradient_accumulation_steps == 0:
                    dist.barrier()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                    self.optimizer.step()
                    self.model.zero_grad()

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
                    
                    if self.rank == 0:
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
                            f"auc {auc:.5f}, m_ap {m_ap:.5f}, lr {lr:e}"
                        )

        if not is_train:
            test_aucs = []
            test_m_aps = []

            vid = ""
            pred_scores = []
            gt_labels = []
            for clip_info in loader.dataset.all_clip_infos:
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

            if self.rank == 0:
                print(f"{split}, loss: {test_loss}, auc {test_auc}, m_ap {test_m_ap}")
                if self.config.tensorboard_writer:
                    self.config.tensorboard_writer.add_scalar(f'{split}/loss', test_loss, epoch)
                    self.config.tensorboard_writer.add_scalar(f'{split}/auc', test_auc, epoch)
                    self.config.tensorboard_writer.add_scalar(f'{split}/m_ap', test_m_ap, epoch)

            self.memory_manager.cleanup(force=True)

            return test_m_ap

def init_models(args):
    """Initialize all models"""
    # lang model
    lang_model = bert_hugface.BertHugface(pretrain_stage=False)
    if os.path.exists(args.lang_pretrain_ckpt_path):
        lang_model.load_state_dict(torch.load(args.lang_pretrain_ckpt_path))

    # vision model
    if args.data_mode == "image":
        if args.model_type == "r50tsm":
            vision_model = resnet50_tsm.Resnet50TSM(segments_size=args.clip_frame_num, shift_div=8, pretrain_stage=False)
        elif args.model_type == "r50":
            vision_model = resnet50.Resnet50(segments_size=args.clip_frame_num, pretrain_stage=False)
        else:
            raise RuntimeError(f"Unknown model_type {args.model_type}")
    else:
        vision_model = resnet50_tsm.Resnet50TSM(segments_size=args.clip_frame_num, shift_div=8, pretrain_stage=False)
    
    if os.path.exists(args.vision_pretrain_ckpt_path):
        vision_model.load_state_dict(torch.load(args.vision_pretrain_ckpt_path))


    # two stream model
    if args.data_mode == "text":
        model = lang_model
        model.build_chapter_head()
    elif args.data_mode == "image":
        model = vision_model
        model.build_chapter_head()
    elif args.data_mode == "all":
        lang_base_model = lang_model.base_model
        vision_base_model = vision_model.base_model
        hidden_size = 128
        model = two_stream_window.TwoStream(
            lang_base_model,
            vision_base_model,
            lang_model.embed_size,
            vision_model.feature_dim,
            args.clip_frame_num,
            hidden_size,
            args.window_size)
        model.build_chapter_head(output_size=2, head_type=args.head_type)
    else:
        raise RuntimeError(f"Unknown data mode {args.data_mode}")
    
    return model

def main_worker(rank, world_size, args):
    """Main worker function for distributed training"""
    setup_ddp(rank, world_size)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Initialize model
    model = init_models(args)

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
        args.img_dir, args.data_file, args.train_vid_file,
        tokenizer, args.clip_frame_num, args.max_text_len,
        window_size=args.window_size, mode=args.data_mode,
        transform=train_vision_preprocess
    )
    test_dataset = InferWindowClipDataset(
        args.img_dir, args.test_clips_json, tokenizer,
        args.clip_frame_num, args.max_text_len,
        window_size=args.window_size,mode=args.data_mode,
        transform=test_vision_preprocess
    )
    
    if rank == 0:
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
    dist.barrier()

    tensorboard_writer = None
    if rank == 0:
        tensorboard_log = os.path.dirname(args.ckpt_path)
        tensorboard_writer = SummaryWriter(tensorboard_log)
        
    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(
        data_mode=args.data_mode,
        max_epochs=args.epoch,
        start_epoch=args.start_epoch,
        best_result=float(args.best_result),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_step,
        learning_rate=args.learning_rate,
        block_size=args.max_text_len,
        lr_decay_type=args.lr_decay_type,
        lr_decay=True,
        warmup_epochs=args.epoch//100,
        final_epochs=args.epoch//100*90, 
        num_workers=args.num_workers,
        ckpt_path=args.ckpt_path,
        tensorboard_writer=tensorboard_writer
    )

    trainer = Trainer(model, train_dataset, test_dataset, tconf, rank, world_size)

    try:
        trainer.train()
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        dist.barrier()
        cleanup_ddp()
        trainer.memory_manager.shutdown()




def run():
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu_id', type=str, default='0,1')
    parser.add_argument('--data_mode', type=str, default="all", help="text (text only), image (image only) or all (multiple-model)")
    parser.add_argument('--model_type', type=str, default="two_stream", help="bert, r50tsm, two_stream")
    parser.add_argument('--clip_frame_num', type=int, default=12)
    parser.add_argument('--epoch', type=int, default=270)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--lr_decay_type', type=str, default="cosine")
    parser.add_argument('--head_type', type=str, default="cross_attn", help="mlp, self_attn, cross_attn, only work on two_stream model")
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_accum_step', type=int, default=4)
    parser.add_argument('--max_text_len', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=2e-6)
    parser.add_argument('--best_result', type=str, default='-inf')

    parser.add_argument('--train_vid_file', type=str, default="./dataset/final_train.txt")
    parser.add_argument('--test_vid_file', type=str, default="./dataset/final_validation.txt")
    parser.add_argument('--img_dir', type=str, default="./dataset/youtube_video_frame_dataset")
    parser.add_argument('--data_file', type=str, default="./dataset/all_in_one_with_subtitle_final.csv")
    parser.add_argument('--test_clips_json', type=str, default="./dataset/validation_clips_clip_frame_num_16.json")
    parser.add_argument('--vision_pretrain_ckpt_path', type=str, default="./checkpoint/r50tsm/batch_64_lr_decay_cosine_train_test_split/pretrain.pth")
    parser.add_argument('--lang_pretrain_ckpt_path', type=str, default="./checkpoint/hugface_bert_pretrain/batch_64_lr_decay_cosine_train_test_split/pretrain.pth")
    parser.add_argument('--ckpt_path', type=str, default="./checkpoint/chapter_localization/")
    
    args = parser.parse_args()
    set_random_seed.use_fix_random_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    world_size = 2

    try:
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e
    finally:
        print(f"Total time: {(time.time()-start_time)/3600}h")

if __name__ == '__main__':
    run()