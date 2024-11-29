"""
train chapter title genration model
"""

import math
import os
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import PegasusTokenizer
from data.youtube_chapter_title_dataset import YoutubeChapterTitleWithVisionEmbDataset
from model.lang import pegasus_vision_emb
from common_utils import set_random_seed


logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    block_size = 512
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.01  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    lr_decay_type = "cosine"
    warmup_epochs = 30 
    final_epoch = 2700 
    # checkpoint settings
    ckpt_path = None
    num_workers = 0    # for DataLoader
    # tensorboard writer
    tensorboard_writer = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, tokenizer, train_dataset, test_dataset, config):
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
        try:
            # DataParallel wrappers keep raw model object in .module attribute
            raw_model = self.model.module if hasattr(self.model, "module") else self.model

            checkpoint_dir = os.path.join(self.config.ckpt_path, f"checkpoint-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")

            checkpoint_dirs = []
            for d in os.listdir(self.config.ckpt_path):
                if d.startswith('checkpoint-') and os.path.isdir(os.path.join(self.config.ckpt_path, d)):
                    checkpoint_dirs.append(d)
            
            if len(checkpoint_dirs) > 10:
                checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
                oldest_dir = os.path.join(self.config.ckpt_path, checkpoint_dirs[0])
                if os.path.exists(oldest_dir):
                    try:
                        for f in os.listdir(oldest_dir):
                            os.remove(os.path.join(oldest_dir, f))
                        os.rmdir(oldest_dir)
                        print(f"Removed old checkpoint: {oldest_dir}")
                    except Exception as e:
                        print(f"Warning: Failed to remove old checkpoint {oldest_dir}: {e}")
            
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                "epoch": epoch,
                "best_result": best_result,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                }, checkpoint_path)

            latest_best_path = os.path.join(self.config.ckpt_path, "checkpoint_best.pth")
            checkpoint_abs_path = os.path.abspath(checkpoint_path)

            if os.path.exists(latest_best_path):
                if os.path.islink(latest_best_path):
                    os.remove(latest_best_path)
                else:
                    os.remove(latest_best_path)
            os.symlink(checkpoint_abs_path, latest_best_path)
            
            print(f"Saved checkpoint at epoch {epoch} with best result {best_result}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            raise


    def train(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.config)

        best_result = float('-inf')
        test_result = float('-inf')
        for epoch in range(self.config.max_epochs):
            train_acc = self.run_epoch('train', epoch)
            if self.test_dataset is not None and epoch % 10 == 0:
                test_result = self.run_epoch('test', epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_result > best_result
            if self.config.ckpt_path is not None and good_model:
                best_result = test_result
                self.save_checkpoint(epoch, best_result)
    

    def run_epoch(self, split, epoch):
        is_train = split == 'train'
        self.model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

        losses = []
        accs = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, (vision_embs, vision_attention_mask, text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids) in pbar:
            vision_embs = vision_embs.to(self.device)
            vision_attention_mask = vision_attention_mask.to(self.device)
            text_ids = text_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)   
            input_decode_ids = input_decode_ids.to(self.device)
            decode_attention_mask = decode_attention_mask.to(self.device)
            target_decode_ids = target_decode_ids.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                logits = self.model(vision_embs, vision_attention_mask, text_ids, attention_mask, decoder_input_ids=input_decode_ids, decoder_attention_mask=decode_attention_mask)

                # calculate loss and acc
                mask = torch.nonzero(decode_attention_mask == 1)
                valid_logits = logits[mask[:, 0], mask[:, 1], :]
                valid_targets = target_decode_ids[mask[:, 0], mask[:, 1]]
                loss = F.cross_entropy(valid_logits.view(-1, valid_logits.size(-1)), valid_targets.view(-1))
                
                # acc
                cpu_y = valid_targets.cpu().numpy()
                topk_scores, topk_labels = valid_logits.data.topk(1, 1, True, True)
                topk_ind = topk_labels.squeeze(1).cpu().numpy()
                correct = np.sum(topk_ind == cpu_y)
                count = len(cpu_y)
                acc = correct / count

                losses.append(loss.item())
                accs.append(acc)
                
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
                self.config.tensorboard_writer.add_scalar('Train/acc', acc, n_iter)

                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, acc {acc:.5f}. lr {lr:e}")

        if not is_train:
            test_loss = float(np.mean(losses))
            test_acc = float(np.mean(accs))
            print("test loss: %f, acc %f"%(test_loss, test_acc))
            self.config.tensorboard_writer.add_scalar('Test/loss', test_loss, epoch)
            self.config.tensorboard_writer.add_scalar('Test/acc', test_acc, epoch)
            return test_acc
        else:
            train_loss = float(np.mean(losses))
            train_acc = float(np.mean(accs))
            return train_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='video chapter title generation model')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    parser.add_argument('--model_type', default="pegasus", type=str)
    parser.add_argument('--fusion_type', default="cross_attn", type=str)
    args = parser.parse_args()

    ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/chapter_title_gen_vision_emb/{args.model_type}_batch_{args.batch_size}"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_final.csv"
    vision_emb_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/youtube_video_vision_emb_clip_frame_num_16"
    train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_train.txt"
    test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_validation.txt"
    tensorboard_log = os.path.dirname(ckpt_path)
    tensorboard_writer = SummaryWriter(tensorboard_log)

    set_random_seed.use_fix_random_seed()
    batch_size = args.batch_size
    num_workers = 8
    chapter_title_text_len = 30
    max_text_len = 512

    # tokenizer and model
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
    model = pegasus_vision_emb.PegasusVisionEmb(reinit_head=True, fusion_type=args.fusion_type).to(args.gpu)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # dataset
    train_dataset = YoutubeChapterTitleWithVisionEmbDataset(vision_emb_dir, data_file, train_vid_file, tokenizer, max_text_len, chapter_title_text_len)
    test_dataset = YoutubeChapterTitleWithVisionEmbDataset(vision_emb_dir, data_file, test_vid_file, tokenizer, max_text_len, chapter_title_text_len)
    
    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=args.epoch, batch_size=batch_size, learning_rate=1e-5, block_size=max_text_len,
                        lr_decay_type=args.lr_decay_type, lr_decay=True, warmup_epochs=args.epoch//100, final_epochs=args.epoch//100*90,
                        num_workers=num_workers, ckpt_path=ckpt_path, tensorboard_writer=tensorboard_writer)
    trainer = Trainer(model, tokenizer, train_dataset, test_dataset, tconf)
    trainer.device = args.gpu
    trainer.train()