import os
import math
import time
import torch
import logging
import argparse
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from sklearn import metrics
from torchvision import transforms
from torch.nn import functional as F
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
# from transformers import get_cosine_schedule_with_warmup

from model.lang import bert_hugface
from model.vision import resnet50_tsm
from model.fusion import two_stream_cw_attn_v2

from common_utils import set_random_seed
from memory_cache_utils import MemoryManager
# from embedding_cache_manager import EmbeddingCacheManager

from data.youtube_dataset import WindowClipIDDataset, custom_collate_fn
from data.infer_youtube_video_dataset import InferWindowClipIDDataset
 

logger = logging.getLogger(__name__)

# LRU 캐시를 사용한 embedding 로딩 최적화
@lru_cache(maxsize=1000)
def cached_load_embedding(embedding_file):
    return np.load(embedding_file)

embedding_thread_pool = ThreadPoolExecutor(max_workers=4)

class TrainerConfig:
    data_mode = "all"
    max_epochs = 300
    start_epoch = 0
    best_result = float('-inf')
    batch_size = 80
    learning_rate = 1e-4
    betas = (0.9, 0.999)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    warmup_ratio = 0.05
    validation_interval = 30
    ckpt_path = None
    num_workers = 4
    tensorboard_writer = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def load_vision_embeddings(embedding_dir, video_ids, clip_start_frames, clip_frame_num=16):
    """
    최적화된 vision embedding 로딩 함수
    """
    def load_single_video_embeddings(video_id, start_frames):
        video_embedding_dir = os.path.join(embedding_dir, video_id)
        clip_embeddings = []
        
        for start_frame in start_frames:
            if start_frame == -1:
                padding_emb = torch.zeros((clip_frame_num, 2048), dtype=torch.float32)
                clip_embeddings.append(padding_emb)
                continue

            embedding_file = os.path.join(video_embedding_dir, f"vision_emb_{start_frame}_{start_frame+16}.npy")
            if os.path.exists(embedding_file):
                emb = cached_load_embedding(embedding_file)
                clip_embeddings.append(torch.tensor(emb, dtype=torch.float32))
            else:
                raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
        
        return torch.stack(clip_embeddings)

    # 병렬 처리로 여러 비디오의 embedding 동시 로딩
    futures = []
    for video_id, start_frames in zip(video_ids, clip_start_frames):
        future = embedding_thread_pool.submit(load_single_video_embeddings, video_id, start_frames)
        futures.append(future)
    
    #print("Waiting for futures to complete...")
    embeddings = [future.result() for future in futures]
    #print("All embeddings loaded successfully")
    return torch.stack(embeddings)
        
class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.memory_manager = MemoryManager()
        self.embedding_cache = {}  # Added embedding cache

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.config)

        # self.scheduler = get_cosine_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=int(num_training_steps * config.warmup_ratio),
        #     num_training_steps=len(train_dataset) // config.batch_size * config.max_epochs
        # )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[2e-4, 1e-4, 5e-5],  # 각 그룹별 max lr
            epochs=config.max_epochs,
            steps_per_epoch=len(train_dataset) // config.batch_size,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
        )

        # self.scheduler = CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=15,  # 첫 주기: 15 epochs
        #     T_mult=2,  # 이후 30, 60 epochs 주기
        #     eta_min=5e-7 # 최소 lr
        # )

    def save_checkpoint(self, epoch, best_result):        
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint_dir = os.path.dirname(self.config.ckpt_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create new checkpoint filename with epoch number
        base_path = os.path.splitext(self.config.ckpt_path)[0]  # Remove .pth extension
        ckpt_path = f"{base_path}_{epoch}_score_{best_result:.4f}.pth"
        logger.info(f"Saving checkpoint at epoch {epoch} to {ckpt_path}")

        try:
            checkpoint = {
                "epoch": epoch,
                "best_result": best_result,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
            }

            torch.save(checkpoint, ckpt_path)
            return ckpt_path

        finally:
            del checkpoint
            self.memory_manager.cleanup(force=True)
    
    # Calculate AUC and MAP metrics
    def calculate_metrics(self, labels, scores):
        if len(labels) > 0:
            fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            m_ap = metrics.average_precision_score(labels, scores)
            return auc, m_ap
        return 0, 0

    # Log metrics to tensorboard
    def log_metrics(self, split, metrics_dict, step):
       if hasattr(self.config, 'tensorboard_writer') and self.config.tensorboard_writer is not None:
           for name, value in metrics_dict.items():
               self.config.tensorboard_writer.add_scalar(f'{split}/{name}', value, step)

    def create_dataloader(self, dataset, is_train):
        print(f"Creating dataloader for {'train' if is_train else 'validation'} dataset")
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_train,
            num_workers=self.config.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
            # dataset,
            # shuffle=is_train,
            # batch_size=self.config.batch_size,
            # num_workers=self.config.num_workers,
            # pin_memory=True,
            # persistent_workers=False,
            # prefetch_factor=1
        )
        print(f"Dataloader created with {len(loader)} batches")
        return loader
    
    def process_train_step(self, binary_prob, labels, loss, it, loader, epoch, pbar):
        """Process training step metrics and logging"""
        with torch.no_grad():
            try:
                scores = binary_prob[:, 1].detach().cpu().numpy()
                batch_labels = labels.cpu().numpy()
                batch_auc, batch_m_ap = self.calculate_metrics(batch_labels, scores)

                n_iter = epoch * len(loader) + it
                current_lr = self.scheduler.get_last_lr()[0]

                metrics_dict = {
                    'lr': current_lr,
                    'loss': loss.item(),
                    'auc': batch_auc,
                    'm_ap': batch_m_ap
                }
                self.log_metrics('Train', metrics_dict, n_iter)

                pbar.set_description(
                    f"[{self.memory_manager.get_status_for_pbar()}] "
                    f"epoch {epoch}: train loss {loss.item():.4f}, "
                    f"auc {batch_auc:.4f}, map {batch_m_ap:.4f}, lr {current_lr:e}"
                )
            finally:
                del scores, batch_labels  # numpy array는 자동 정리되지만 명시적으로 처리

    def process_val_step(self, binary_prob, it, batch_size, dataset, loss, pbar):
        """Optimized validation step"""
        with torch.no_grad():
            start_idx = it * batch_size
            end_idx = start_idx + binary_prob.shape[0]
            # 바로 CPU로 이동하고 numpy로 저장
            self.all_predictions[start_idx:end_idx] = binary_prob[:, 1].cpu().numpy()

            pbar.set_description(
                f"[{self.memory_manager.get_status_for_pbar()}] "
                f"epoch: val loss {loss.item():.4f}"
            )

    
    def calculate_epoch_metrics(self, dataset, losses, epoch):
        """Optimized metric calculation"""
        try:
            test_aucs = []
            test_m_aps = []
            
            # 미리 계산된 인덱스로 빠르게 메트릭 계산
            for vid_info in self.video_indices.values():
                pred_scores = self.all_predictions[vid_info["indices"]]
                auc, m_ap = self.calculate_metrics(vid_info["labels"], pred_scores)
                test_aucs.append(auc)
                test_m_aps.append(m_ap)

            test_loss = float(np.mean(losses))
            test_auc = float(np.mean(test_aucs))
            test_m_ap = float(np.mean(test_m_aps))

            print(f"val_loss: {test_loss}, auc: {test_auc}, m_ap: {test_m_ap}")

            metrics_dict = {
                'loss': test_loss,
                'auc': test_auc,
                'm_ap': test_m_ap
            }
            self.log_metrics('infer_test', metrics_dict, epoch)

            return test_m_ap

        finally:
            # validation 관련 데이터 정리
            if hasattr(self, 'all_predictions'):
                del self.all_predictions
            if hasattr(self, 'video_indices'):
                del self.video_indices
            # self.memory_manager.cleanup(force=True)
    
    
    def run_epoch(self, split, epoch, dataset):
        print(f"Starting {split} epoch {epoch}")
    
        max_retries = 3
        retry_count = 0
        is_train = split == 'train'

        while retry_count < max_retries:
            loader = None
            try:    
                if retry_count > 0:
                    logger.info(f"Retry attempt {retry_count}/{max_retries}")
                    self.memory_manager.cleanup(force=True)
                
                self.model.train(is_train)
                loader = self.create_dataloader(dataset, is_train)
                losses = []
                print("Dataloader created successfully")
                if not is_train:
                    # validation 전처리
                    self.video_indices = {}
                    for i, clip_info in enumerate(dataset.all_clip_infos):
                        vid = clip_info["vid"]
                        if vid not in self.video_indices:
                            self.video_indices[vid] = {"indices": [], "labels": []}
                        self.video_indices[vid]["indices"].append(i)
                        self.video_indices[vid]["labels"].append(clip_info["clip_label"])
                    
                    # 결과 저장용 array 미리 할당
                    self.all_predictions = np.zeros(len(dataset))
                    self.memory_manager.cleanup(force=True)
                
                pbar = tqdm(enumerate(loader), total=len(loader))
                device = next(self.model.parameters()).device
                
                for it, batch_data in pbar:
                    try:
                        img_clips, text_ids, attention_masks, labels, clip_info = batch_data

                        # Move other data to device
                        text_ids = text_ids.to(device)
                        attention_masks = attention_masks.to(device)
                        labels = labels.to(device)

                        video_ids = clip_info.pop("video_id")
                        clip_info = {k: v.to(device) for k, v in clip_info.items()}

                        # print(clip_info)

                        # Load Vision Embeddings using clip_start_frame
                        clip_start_frames = clip_info["clip_start_frame"].cpu().numpy().tolist()
                        vision_embeddings = load_vision_embeddings(
                            embedding_dir,
                            video_ids,
                            clip_start_frames,
                            #embedding_cache=self.embedding_cache
                        ).to(device)
                        # print(f'loaded vision_embedding size: {vision_embeddings.shape}')  # [batch_size, num_clips, num_frames, 2048]

                        # Forward pass
                        with torch.set_grad_enabled(is_train):
                            binary_logits, binary_prob = self.model(vision_embeddings, text_ids, attention_masks, clip_info)
                            loss = F.cross_entropy(binary_logits, labels)

                        if is_train:
                            # Training step
                            self.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                            self.optimizer.step()

                            self.process_train_step(binary_prob, labels, loss, it, loader, epoch, pbar)
                        
                        else:
                            # Validation step - store predictions and loss
                            self.process_val_step(binary_prob, it, self.config.batch_size, dataset, loss, pbar)
                            losses.append(loss.item())

                        del binary_logits, binary_prob, loss
                        del img_clips, text_ids, attention_masks, labels, clip_info, vision_embeddings
                    
                    except RuntimeError as e:
                        error_msg = f"RuntimeError in {'training' if is_train else 'validation'} batch {it}"
                        logger.error(f"{error_msg}: {str(e)}")
                        logger.error(f"Batch info: Size={loader.batch_size}, Device={device}")
                        logger.error("Full traceback:", exc_info=True)
                        
                        if "out of memory" in str(e):
                            logger.warning(f"OOM in batch {it}. Attempting recovery...")
                            self.memory_manager.handle_oom()
                            continue
                        raise RuntimeError(f"{error_msg}: {str(e)}")
                    
                    except Exception as e:
                        error_msg = f"Unexpected error in {'training' if is_train else 'validation'} batch {it}"
                        logger.error(f"{error_msg}: {type(e).__name__}: {str(e)}")
                        logger.error("Full traceback:", exc_info=True)
                        raise type(e)(f"{error_msg}: {str(e)}")
                if is_train:
                    self.scheduler.step()
                if not is_train:
                    return self.calculate_epoch_metrics(dataset, losses, epoch)
                return None

            except Exception as e:
                retry_count += 1
                error_location = "training" if is_train else "validation"
                error_msg = (f"Error during {error_location} epoch {epoch} "
                            f"(attempt {retry_count}/{max_retries})")
                
                logger.error(f"{error_msg}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                logger.error("Full traceback:", exc_info=True)
                
                if retry_count == max_retries:
                    raise type(e)(f"{error_msg}: {str(e)}")
                
                self.memory_manager.handle_oom()

            finally:
                if loader is not None:
                    self.memory_manager.cleanup_dataloader(loader)

    def train(self):
        best_result = self.config.best_result
        test_result = float('-inf')

        try:
            for epoch in range(self.config.start_epoch+1, self.config.max_epochs+1):
                # Training
                self.run_epoch('train', epoch, self.train_dataset)
                
                if self.test_dataset is not None and epoch % self.config.validation_interval == 0:
                    # self.memory_manager.cleanup(force=True)
                    test_result = self.run_epoch("infer_test", epoch, self.test_dataset)

                    # Validation 후에만 체크포인트 저장 여부 결정
                    if test_result > best_result:
                        best_result = test_result
                        if self.config.ckpt_path is not None:
                            checkpoint_path = self.save_checkpoint(epoch, best_result)
                            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
                    
                    if hasattr(self.config, 'tensorboard_writer') and self.config.tensorboard_writer is not None:
                        self.config.tensorboard_writer.flush()

                if epoch % 5 == 0:
                    
                    self.memory_manager.cache_manager.clear_cache()
                    self.memory_manager.cleanup(force=True)
                
        except Exception as e:
            # 에러 발생 시 메모리 정리
            self.memory_manager.handle_oom()
            raise e

        finally:
            self.memory_manager.log_memory_stats("Final memory status")
            self.memory_manager.cleanup(force=True)
            if embedding_thread_pool:
                embedding_thread_pool.shutdown()

    
def parse_args():
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--gpu_ids', default='0,1,2,3,4', type=str,help='gpu ids to use (e.g., "0,1,2"). -1 for CPU')
    parser.add_argument('--data_mode', default="all", type=str)
    parser.add_argument('--model_type', default="two_stream", type=str)
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=80, type=int)
    parser.add_argument('--window_size', default=1, type=int)
    parser.add_argument('--max_text_len', default=100, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Paths
    vision_pretrain_ckpt_path = '/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/chapter_localization/head_mlp_batch_16/checkpoint.pth'
    lang_pretrain_ckpt_path = '/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/chapter_localization/head_mlp_batch_16/checkpoint.pth'
    #ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/chapter_localization/cw_attn_batch_{args.batch_size}_frame_{args.clip_frame_num}/checkpoint.pth"
    ckpt_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/checkpoint/chapter_localization/simple_window_batch_{args.batch_size}_frame_{args.clip_frame_num}_v3/checkpoint.pth"
    img_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/youtube_video_frame_dataset"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_final.csv"
    test_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/validation_clips_clip_frame_num_{args.clip_frame_num}.json"
    train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_train.txt"
    test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/final_validation.txt"
    tensorboard_log = os.path.dirname(ckpt_path)
    tensorboard_writer = SummaryWriter(tensorboard_log)
    
    # # Debgging dataset
    # test_clips_json = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/debugging_val_clips_clip_frame_num_{args.clip_frame_num}.json"
    # train_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/debugging_train.txt"
    # test_vid_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/debugging_validation.txt"

    # 임베딩 path
    embedding_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_generation/youtube_video_vision_emb_clip_frame_num_16" 

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

    memory_manager = MemoryManager()
    logger.info("Starting model initialization and data loading...")

    try:
        set_random_seed.use_fix_random_seed()

        # init model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        lang_model = bert_hugface.BertHugface(pretrain_stage=False)
        vision_model = resnet50_tsm.Resnet50TSM(
            segments_size=args.clip_frame_num,
            shift_div=8,
            pretrain_stage=False
        )

        # 언어 모델 체크포인트 로드 및 파라미터 고정
        if os.path.exists(lang_pretrain_ckpt_path):
            checkpoint = torch.load(lang_pretrain_ckpt_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)  # state_dict 추출

            # 모델 로드 시 누락된 키 무시
            missing_keys = lang_model.load_state_dict(state_dict, strict=False).missing_keys
            if missing_keys:
                logger.warning(f"Missing keys for lang model: {missing_keys}")
            
            # 모델 파라미터 고정 (freeze)
            for param in lang_model.parameters():
                param.requires_grad = False
            
            logger.info("Successfully loaded and froze weights for lang model.")

        # 비전 모델 체크포인트 로드 및 파라미터 고정
        if os.path.exists(vision_pretrain_ckpt_path):
            checkpoint = torch.load(vision_pretrain_ckpt_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)  # state_dict 추출

            # 모델 로드 시 누락된 키 무시
            missing_keys = vision_model.load_state_dict(state_dict, strict=False).missing_keys
            if missing_keys:
                logger.warning(f"Missing keys for vision model: {missing_keys}")
            
            # 모델 파라미터 고정 (freeze)
            for param in vision_model.parameters():
                param.requires_grad = False
            
            logger.info("Successfully loaded and froze weights for vision model.")
            # del lang_state_dict, vision_state_dict
            # memory_manager.cleanup(force=True)

        # GlobalTwoStream model
        if args.data_mode == "all":
            logger.info("Initializing TwoStream model...")
            hidden_size = 128
            lang_base_model = lang_model.base_model
            vision_base_model = vision_model.base_model

            model = two_stream_cw_attn_v2.TwoStream(
                lang_model=lang_base_model,
                vision_model=vision_base_model,
                lang_embed_size=lang_model.embed_size,
                vision_embed_size=vision_model.feature_dim,
                segment_size=args.clip_frame_num,
                hidden_size=hidden_size,
                window_size = args.window_size
            )

            memory_manager.cleanup(force=True)

            model.build_chapter_head(output_size=2)
            model = model.to(device)
            if torch.cuda.is_available() and len(gpu_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        else:
            raise RuntimeError(f"Unknown data mode {args.data_mode}")

        # dataset
        logger.info("Initializing datasets...")
        train_vision_preprocess = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter()], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_vision_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Initializing datasets...")

        train_dataset = WindowClipIDDataset(
            img_dir=img_dir,
            data_file=data_file,
            vid_file=train_vid_file,
            tokenizer=tokenizer,
            clip_frame_num=args.clip_frame_num,
            max_text_len=args.max_text_len,
            window_size=args.window_size,
            mode=args.data_mode,
            transform=train_vision_preprocess
        )
        print("Train dataset initialized")

        test_dataset = InferWindowClipIDDataset(
            img_dir=img_dir,
            json_paths=test_clips_json,
            tokenizer=tokenizer,
            clip_frame_num=args.clip_frame_num,
            max_text_len=args.max_text_len,
            window_size=args.window_size,
            mode=args.data_mode,
            transform=test_vision_preprocess
        )
        print("Test dataset initialized")

        print("Initializing trainer...")

        # 데이터셋 로딩 후 메모리 정리
        memory_manager.cleanup(force=True)

        # initialize a trainer instance and kick off training
        logger.info("Initializing trainer...")
        tconf = TrainerConfig(
            data_mode=args.data_mode,
            max_epochs=args.epoch,
            start_epoch = 0,
            best_result = float('-inf'),
            batch_size=args.batch_size,
            learning_rate=1e-4,
            betas = (0.9, 0.999),
            grad_norm_clip = 1.0,
            weight_decay = 0.1,
            warmup_ratio = 0.05,
            validation_interval = 30,
            num_workers=4,
            ckpt_path=ckpt_path,
            tensorboard_writer=tensorboard_writer
        )

        print("Trainer config created")

        trainer = Trainer(model, train_dataset, test_dataset, tconf)
        print("Trainer initialized")

        trainer.device = device
        print(f"Trainer device set to {device}")

        print("Starting training...")

        logger.info("Starting training...")
        trainer.train()

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

    finally:
        # 최종 정리
        memory_manager.shutdown()
        embedding_thread_pool.shutdown()