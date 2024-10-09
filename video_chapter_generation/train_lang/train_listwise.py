"""
Use hugface transformer pretrained language model and finetune on youtube subtitle dataset

"""

import math
import os
import logging
from random import randint

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from transformers import OpenAIGPTTokenizer, BertTokenizer
from data.youtube_dataset import YoutubeListwiseClipDataset
from data.infer_youtube_video_dataset import InferYoutubeVideoDataset
from model.lang import bert_hugface_listnet
from common_utils import set_random_seed


logger = logging.getLogger(__name__)

class TrainerConfig:
    # data mode
    data_mode = "text"
    # model type
    model_type = "gpt"
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
    warmup_epochs = 200 
    final_epochs = 2500 
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

        best_result = float('-inf')
        test_result = float('-inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(self.config.max_epochs):
            self.run_epoch('train', epoch, self.train_dataset)

            if self.test_dataset is not None and epoch % 5 == 0:
                infer_train_dataset = self.test_dataset["infer_train"]
                infer_test_dataset = self.test_dataset["infer_test"]

                # infer_train_result = self.run_epoch("infer_train", epoch, infer_train_dataset)
                infer_test_result = self.run_epoch("infer_test", epoch, infer_test_dataset)
                test_result = infer_test_result

            # supports early stopping based on the test acc, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_result > best_result
            if self.config.ckpt_path is not None and good_model:
                best_result = test_result
                self.save_checkpoint()
    

    def run_epoch(self, split, epoch, dataset):
        is_train = split == 'train'
        self.model.train(is_train)

        if is_train:
            run_time = 1
        else:
            # run_time = 500       # for test, we run multiple different videos
            run_time = len(dataset.vids)

        losses = []
        test_aucs = []
        test_m_aps = []

        for i in range(run_time):
            if not is_train:
                # dataset.random_choose_vid()
                dataset.manual_choose_vid(dataset.vids[i])
                shuffle = False
            else:
                shuffle = True

            # for calculate map in testing
            pred_scores = []
            gt_labels = []

            loader = DataLoader(dataset, shuffle=shuffle, pin_memory=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (img_clip, text_ids, attention_mask, label) in pbar:
                img_clip = img_clip.float().to(self.device)
                text_ids = text_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)   
                label = label.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    if self.config.data_mode == "text":
                        if is_train:
                            batch_size, slate_length, text_length = text_ids.size()
                            pos_num = int(batch_size // 2)
                            neg_num = batch_size - pos_num
                            balance_selected_indices = [slate_length*ix + randint(0, 1) for ix in range(pos_num)] + [slate_length*ix + randint(2, slate_length-1) for ix in range(pos_num, batch_size)]
                            # balance_selected_indices = [slate_length*ix for ix in range(pos_num)] + [slate_length*ix + randint(1, slate_length-1) for ix in range(pos_num, batch_size)]
                            balance_selected_indices = torch.from_numpy(np.array(balance_selected_indices)).long().to(self.device)   

                            binary_cls_label = [1 for ix in range(pos_num)] + [0 for ix in range(neg_num)]
                            binary_cls_label = torch.from_numpy(np.array(binary_cls_label)).long().to(self.device) 
                            output = self.model.train_forward(text_ids, attention_mask, label, balance_selected_indices, binary_cls_label)
                        else:
                            output = self.model.test_forward(text_ids, attention_mask, targets=label)
                    else:
                        raise RuntimeError(f"have not yet implemented data mode {self.config.data_mode}")
                    
                    binary_prob = output["binary_prob"]
                    loss = output["loss"]
                    losses.append(loss.item())

                    # record results
                    if is_train:
                        cpu_y = list(binary_cls_label.cpu().numpy())
                    else:
                        cpu_y = list(label.cpu().numpy())
                    scores = binary_prob[:, 1]
                    scores = scores.detach().cpu().numpy()
                    gt_labels.extend(cpu_y)
                    pred_scores.extend(scores)
                    
                if is_train:
                    # record training results
                    fpr, tpr, thresholds = metrics.roc_curve(cpu_y, scores, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    m_ap = metrics.average_precision_score(cpu_y, scores)

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
                    self.config.tensorboard_writer.add_scalar('Train/auc', auc, n_iter)
                    self.config.tensorboard_writer.add_scalar('Train/m_ap', m_ap, n_iter)
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, auc {auc:.5f}, m_ap {m_ap:.5f}, lr {lr:e}")

            if not is_train:
                fpr, tpr, thresholds = metrics.roc_curve(gt_labels, pred_scores, pos_label=1)
                test_auc = metrics.auc(fpr, tpr)
                test_m_ap = metrics.average_precision_score(gt_labels, pred_scores)
                test_aucs.append(test_auc)
                test_m_aps.append(test_m_ap)

        if not is_train:
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
    parser.add_argument('--gpu', default=5, type=int)
    parser.add_argument('--model_type', default="bert", type=str)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr_decay_type', default="cosine", type=str)
    args = parser.parse_args()

    pretrained_ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/hugface_{args.model_type}_pretrain/batch_64_lr_decay_cosine_train_test_split/pretrain.pth"
    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/video_chapter_hugface_{args.model_type}/batch_{args.batch_size}_lr_decay_{args.lr_decay_type}_cls_on_listwise_loader2/checkpoint.pth"
    img_dir = "/opt/tiger/youtube_video_frame_dataset"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    train_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/train.txt"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"
    tensorboard_log = os.path.dirname(ckpt_path)
    tensorboard_writer = SummaryWriter(tensorboard_log)

    set_random_seed.use_fix_random_seed()
    batch_size = args.batch_size
    data_mode = "text"  # text (text only), image (image only) or all (multiple-model)
    clip_frame_num = 8
    num_workers = 8
    max_text_len = 50

    # tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = bert_hugface_listnet.BertHugface(pretrain_stage=False)

    # load pretrained model
    model.load_state_dict(torch.load(pretrained_ckpt_path))
    # model.load_constrast_checkpoint(pretrained_ckpt_path)
    model.build_chapter_head()
    # model.fix_backbone()
    model = model.to(args.gpu)
    
    # dataset
    # train_dataset = YoutubeClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, mode=data_mode)
    train_dataset = YoutubeListwiseClipDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, negative_clip_num=10, mode=data_mode)
    
    infer_dataset = {
        "infer_train": InferYoutubeVideoDataset(img_dir, data_file, train_vid_file, tokenizer, clip_frame_num, max_text_len, mode=data_mode),
        "infer_test": InferYoutubeVideoDataset(img_dir, data_file, test_vid_file, tokenizer, clip_frame_num, max_text_len, mode=data_mode)
    }
    
    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(data_mode=data_mode, model_type=args.model_type, max_epochs=args.epoch, batch_size=batch_size, learning_rate=1e-5, block_size=max_text_len,
                        lr_decay_type=args.lr_decay_type, lr_decay=True, warmup_epochs=args.epoch//10, final_epochs=args.epoch//10*9,
                        num_workers=num_workers, ckpt_path=ckpt_path, tensorboard_writer=tensorboard_writer)
    trainer = Trainer(model, tokenizer, train_dataset, infer_dataset, tconf)
    trainer.device = args.gpu
    trainer.train()






