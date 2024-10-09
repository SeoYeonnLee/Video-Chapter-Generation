"""
Use hugface transformer pretrained language model and finetune on youtube subtitle dataset

Before run, you should config accelerate by
accelerate config
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
Which type of machine are you using? ([0] No distributed training, [1] multi-GPU, [2] TPU): 1
How many different machines will you use (use more than 1 for multi-node training)? [1]: 
How many processes in total will you use? [1]: 8
Do you wish to use FP16 (mixed precision)? [yes/NO]: yes


run command: accelerate launch train_chapter_title_gen_accelerator.py
Some bugs here, training procedure will stuck


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
from data.youtube_chapter_title_dataset import YoutubeChapterTitleDataset
from model.lang import pegasus_hugface
from common_utils import set_random_seed
from accelerate import Accelerator


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
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0    # for DataLoader
    # tensorboard writer
    tensorboard_writer = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, tokenizer, train_dataset, test_dataset, config, accelerator):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.accelerator = accelerator
        self.device = self.accelerator.device

        self.train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
        self.test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader = self.accelerator.prepare(self.test_loader)
        

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.config.ckpt_path), exist_ok=True)
        print("saving %s" % self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), self.config.ckpt_path)

    def train(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.config)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.model = self.accelerator.prepare(self.model)

        best_loss = float('inf')
        test_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(self.config.max_epochs):
            self.accelerator.wait_for_everyone()
            self.run_epoch('train', epoch)
            if self.test_dataset is not None and epoch % 20 == 0:
                test_loss = self.run_epoch('test', epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()
    

    def run_epoch(self, split, epoch):
        is_train = split == 'train'
        self.model.train(is_train)
        loader = self.train_loader if is_train else self.test_loader

        losses = []
        accs = []
        pbar = tqdm(enumerate(loader), total=len(loader), disable=not self.accelerator.is_local_main_process) if is_train else enumerate(loader)
        for it, (text_ids, attention_mask, decode_text_ids, decode_attention_mask) in pbar:
            text_ids = text_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)   
            decode_text_ids = decode_text_ids.to(self.device)
            decode_attention_mask = decode_attention_mask.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                logits = self.model(text_ids, attention_mask, decoder_input_ids=decode_text_ids, decoder_attention_mask=decode_attention_mask)

                # calculate loss and acc
                mask = torch.nonzero(decode_attention_mask == 1)
                valid_logits = logits[mask[:, 0], mask[:, 1], :]
                valid_targets = decode_text_ids[mask[:, 0], mask[:, 1]]
                loss = F.cross_entropy(valid_logits.view(-1, valid_logits.size(-1)), valid_targets.view(-1))
                
                # acc
                gather_loss, gather_valid_logits, gather_valid_targets = self.accelerator.gather((loss.unsqueeze(dim=0), valid_logits, valid_targets))

                if self.accelerator.is_local_main_process:
                    loss_m = torch.mean(gather_loss)
                    cpu_y = gather_valid_targets.cpu().numpy()
                    topk_scores, topk_labels = gather_valid_logits.data.topk(1, 1, True, True)
                    topk_ind = topk_labels.squeeze(1).cpu().numpy()
                    correct = np.sum(topk_ind == cpu_y)
                    count = len(cpu_y)
                    acc = correct / count
                    print(f"loss {loss_m.item()}")
                    losses.append(loss_m.item())
                    accs.append(acc)
                
            if is_train:
                # backprop and update the parameters
                self.model.zero_grad()
                self.accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                # decay the learning rate based on our progress
                if self.config.lr_decay:
                    self.tokens += (attention_mask > 0).sum()

                    if self.tokens < self.config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, self.config.warmup_tokens))
                    else:
                        progress = float(self.tokens - self.config.warmup_tokens) / float(max(1, self.config.final_tokens - self.config.warmup_tokens))
                        # cosine learning rate decay
                        if self.config.lr_decay_type == "cosine":
                            # lr_mult = max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr_mult = max(0.001, 0.5 * (1.0 + math.cos(math.pi * progress)))   # this more proper

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
                if self.accelerator.is_local_main_process:
                    self.config.tensorboard_writer.add_scalar('Train/loss', loss_m.item(), n_iter)
                    self.config.tensorboard_writer.add_scalar('Train/acc', acc, n_iter)
                    
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        if not is_train and self.accelerator.is_local_main_process:
            test_loss = float(np.mean(losses))
            test_acc = float(np.mean(accs))
            print("test loss: %f, acc %f"%(test_loss, test_acc))
            self.config.tensorboard_writer.add_scalar('Test/loss', test_loss, epoch)
            self.config.tensorboard_writer.add_scalar('Test/acc', test_acc, epoch)
            return test_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='video chapter title generation model')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--epoch', default=3000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    args = parser.parse_args()


    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/chapter_title_hugface_pegasus/batch_{args.batch_size}_lr_decay_{args.lr_decay_type}/checkpoint.pth"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"

    accelerator = Accelerator()
    tensorboard_writer = None
    if accelerator.is_local_main_process:
        tensorboard_log = os.path.dirname(ckpt_path)
        tensorboard_writer = SummaryWriter(tensorboard_log)

    set_random_seed.use_fix_random_seed()
    batch_size = args.batch_size
    num_workers = 16
    max_text_len = 512
    chapter_title_text_len = 30

    # tokenizer and model
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
    model = pegasus_hugface.PegasusHugface(reinit_head=True)
    
    # dataset
    train_dataset = YoutubeChapterTitleDataset(data_file, train_vid_file, tokenizer, max_text_len, chapter_title_text_len)
    test_dataset = YoutubeChapterTitleDataset(data_file, test_vid_file, tokenizer, max_text_len, chapter_title_text_len)
    
    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=args.epoch, batch_size=batch_size, learning_rate=1e-5, block_size=max_text_len,
                        lr_decay_type=args.lr_decay_type, lr_decay=True, warmup_tokens=args.epoch//100*len(train_dataset)*max_text_len//2, final_tokens=args.epoch//5*4*len(train_dataset)*max_text_len//2,
                        num_workers=num_workers, ckpt_path=ckpt_path, tensorboard_writer=tensorboard_writer)
    trainer = Trainer(model, tokenizer, train_dataset, test_dataset, tconf, accelerator)
    trainer.train()






