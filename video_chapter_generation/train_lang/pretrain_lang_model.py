 """
pretrain language model

"""
import math
import os
# os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4,5,6"
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.lang.gpt import GPT, GPTConfig
from data.youtube_subtitle_dataset import YoutubeClipSubtitleGloveDataset, YoutubeClipSubtitleDataset


logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
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

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        # self.device = 'cpu'
        # if torch.cuda.is_available():
            # self.device = torch.cuda.current_device()
            # self.model = self.model.to(self.device)
            # self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        os.makedirs(os.path.dirname(self.config.ckpt_path), exist_ok=True)
        # logger.info("saving %s", self.config.ckpt_path)
        print("saving %s" % self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.config)

        best_loss = float('inf')
        test_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(self.config.max_epochs):
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
        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)

        losses = []
        accs = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, (x, y) in pbar:

            # place data on the correct device
            x = x.to(self.device)
            y = y.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                logits, loss = self.model(x, y)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

                # acc
                mask = torch.nonzero(y != -1)
                valid_logits = logits[mask[:, 0], mask[:, 1], :]
                valid_y = y[mask[:, 0], mask[:, 1]]

                cpu_y = valid_y.cpu().numpy()
                topk_scores, topk_labels = valid_logits.data.topk(1, 1, True, True)
                topk_ind = topk_labels.squeeze(1).cpu().numpy()
                correct = np.sum(topk_ind == cpu_y)
                count = len(cpu_y)
                acc = correct / count
                accs.append(acc)
                
            if is_train:

                # backprop and update the parameters
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                # decay the learning rate based on our progress
                if self.config.lr_decay:
                    self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < self.config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, self.config.warmup_tokens))
                    else:
                        progress = float(self.tokens - self.config.warmup_tokens) / float(max(1, self.config.final_tokens - self.config.warmup_tokens))
                        # cosine learning rate decay
                        if self.config.lr_decay_type == "cosine":
                            # lr_mult = max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr_mult = max(0.001, 0.01 * (1.0 + math.cos(math.pi * progress)))   # this more proper

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
                self.config.tensorboard_writer.add_scalar('Loss/train', loss.item(), n_iter)
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        if not is_train:
            test_loss = float(np.mean(losses))
            test_acc = float(np.mean(accs))
            # logger.info("test loss: %f", test_loss)
            print("test loss: %f, acc: %f"%(test_loss, test_acc))
            self.config.tensorboard_writer.add_scalar('Loss/test', test_loss, epoch)
            self.config.tensorboard_writer.add_scalar('Acc/test', test_acc, epoch)
            return test_loss




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='pretrain language model')
    parser.add_argument('--gpu', default=6, type=int)
    parser.add_argument('--model_config', default='temp')
    parser.add_argument('--epoch', default=3000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--layer_num', default=12, type=int)
    parser.add_argument('--learn_pe', action='store_true')
    parser.add_argument('--glove_emb', action='store_true')
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    args = parser.parse_args()


    model_config = "layer"+str(args.layer_num)+"_batch"+str(args.batch_size)
    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/gpt_pretrain_using_glove/{model_config}_learn_pe_{args.learn_pe}_glove_emb_{args.glove_emb}_lr_decay_{args.lr_decay_type}/gpt_pretrain.pth"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    glove_pickle_file = "/opt/tiger/video_chapter_generation/glove.840B.300d.pickle"
    tensorboard_log = os.path.dirname(ckpt_path)
    tensorboard_writer = SummaryWriter(tensorboard_log)

    # use glove embedding or not
    if args.glove_emb:
        vocab_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/vocab_in_glove.txt"
        dataset = YoutubeClipSubtitleGloveDataset(data_file, glove_pickle_file, vocab_file, clip_frame_num=8, max_text_len=50)
    else:
        vocab_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/vocab.txt"
        dataset = YoutubeClipSubtitleDataset(data_file, vocab_file, clip_frame_num=8, max_text_len=50)

    vocab_size = dataset.vocab_size
    block_size = dataset.max_text_len
    mconf = GPTConfig(vocab_size=vocab_size, block_size=block_size, using_pretrained_embed=args.glove_emb, learnable_pos_emb=args.learn_pe, n_layer=args.layer_num, n_head=10, n_embd=300)
    model = GPT(mconf).to(args.gpu)


    # initialize a trainer instance and kick off training
    batch_size = args.batch_size
    tconf = TrainerConfig(max_epochs=args.epoch, batch_size=batch_size, learning_rate=1e-3,
                        lr_decay_type=args.lr_decay_type, lr_decay=True, warmup_tokens=args.epoch//100*len(dataset)*block_size, final_tokens=args.epoch//10*9*len(dataset)*block_size,
                        num_workers=8, ckpt_path=ckpt_path, tensorboard_writer=tensorboard_writer)
    trainer = Trainer(model, dataset, dataset, tconf)
    trainer.device = args.gpu
    trainer.train()




