import gc
import math
import torch
import logging

from torch import nn
from torch.nn import functional as F
from einops import rearrange

from memory_cache_utils import MemoryManager
from model.fusion.window_self_attention_simple import VideoChapterClassifier

logger = logging.getLogger(__name__)

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Layer norms
        self.lang_norm = nn.LayerNorm(hidden_size)
        self.vision_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)

        # Attention projections
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Position encodings
        self.frame_pos_encoding = nn.Sequential(
            nn.Linear(1, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

        # Global position encoding for both frames and text
        self.global_pos_encoding = nn.Sequential(
            nn.Linear(2, hidden_size//2),  # [timestamp, duration]
            nn.LayerNorm(hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

        # Vision information pooling
        self.vision_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Learnable fusion ratio
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize weights
        scale = 1.0 / math.sqrt(self.head_dim)
        # attention projection layers, bias
        for module in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=scale)
            nn.init.zeros_(module.bias)

        # Position encoding layers
        for module in [self.frame_pos_encoding, self.global_pos_encoding]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        for module in [self.vision_pool, self.ffn]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def get_frame_positions(self, clip_start_frame, num_frames, total_frames):
        """Calculate both local and global positions for frames
        Args:
            clip_start_frame: 클립의 시작 프레임 인덱스
            num_frames: 클립당 고정된 프레임 수
            total_frames: 비디오의 총 프레임 수
        """
        device = clip_start_frame.device

        # Local positions (0-1 within clip)
        local_positions = torch.arange(num_frames, device=device).float()
        local_positions = local_positions / (num_frames - 1)
        local_positions = torch.clamp(local_positions, 0, 1)
        
        # Global positions
        frame_indices = torch.arange(clip_start_frame, clip_start_frame + num_frames, device=device)
        frame_indices = torch.clamp(frame_indices.float(), 0, total_frames-1)

        # Apply log-scale normalization
        eps = 1e-6
        log_numerator = torch.log1p(frame_indices)
        log_denominator = torch.log1p(total_frames)
        normalized_pos = log_numerator / (log_denominator + eps)
        normalized_pos = torch.clamp(normalized_pos, 0, 1)
    
         # Scale factor to indicate relative video length
        max_frames = 1800
        length_factor = torch.log1p(total_frames) / torch.log1p(torch.tensor(max_frames, device=device))
        length_factor = torch.clamp(length_factor, 0, 1)

        global_info = torch.stack([normalized_pos, torch.full_like(frame_indices, length_factor)], dim=-1)
        
        return local_positions.unsqueeze(-1), global_info

    def forward(self, lang_out, vision_out, clip_info):
        """
        Args:
            lang_out: [batch, 1, hidden_size]
            vision_out: [batch, num_frame, hidden_size]
            clip_info: {
                'clip_start_frame': [batch, num_frame],
                'total_frames': [batch],
                'target_clip_idx': [batch],
                'total_num_clips': [batch]
            }
        """
        device = lang_out.device
        if not hasattr(self, '_moved_to_device'):
            self = self.to(device)
            self._moved_to_device = True

        batch_size, num_frame = vision_out.shape[:2]

        # Get positions for each batch
        all_local_pos = []
        all_global_pos = []
        for i in range(batch_size):
            local_pos, global_pos = self.get_frame_positions(
                clip_info['clip_start_frame'][i], 
                num_frame,
                clip_info['total_frames'][i],
            )
            all_local_pos.append(local_pos)
            all_global_pos.append(global_pos)
        
        local_positions = torch.stack(all_local_pos, dim=0)  # [batch, num_frame, 1]
        global_positions = torch.stack(all_global_pos, dim=0)  # [batch, num_frame, 2]

        # Generate position embeddings
        local_pos_emb = self.frame_pos_encoding(local_positions)
        global_pos_emb = self.global_pos_encoding(global_positions)

        # Add position embeddings to vision features
        vision_out = vision_out + local_pos_emb + global_pos_emb

        # Add global position to text features
        clip_progress = clip_info['target_clip_idx'].float() / clip_info['total_num_clips'].float()
        text_global_pos = torch.stack([clip_progress, clip_info['total_frames'].float() / 1800.0], dim=-1).unsqueeze(1).to(device)  # [batch, 1, 2]  # Normalize by max frames
        
        text_pos_emb = self.global_pos_encoding(text_global_pos)
        lang_out = lang_out + text_pos_emb

        # Layer normalization
        normed_lang = self.lang_norm(lang_out)  # [batch, 1, hidden_size]
        normed_vision = self.vision_norm(vision_out)  # [batch, num_frame, hidden_size]
        
        # Project to Q, K, V
        query = self.query_proj(normed_lang)  # [batch, 1, hidden_size]
        key = self.key_proj(normed_vision)    # [batch, num_frame, hidden_size]
        value = self.value_proj(normed_vision) # [batch, num_frame, hidden_size]

        # Reshape for multi-head attention
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        key = key.view(batch_size, num_frame, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, num_frame, head_dim]
        value = value.view(batch_size, num_frame, self.num_heads, self.head_dim).transpose(1, 2) # [batch, num_heads, num_frame, head_dim]

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) # [batch, num_heads, 1, num_frame]
        attn_scores = torch.clamp(attn_scores, -10, 10)
        attn_scores = attn_scores / (scale + 1e-6)
        
        max_score = torch.max(attn_scores, dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(attn_scores - max_score)
        attn_probs = exp_scores / (torch.sum(exp_scores, dim=-1, keepdim=True) + 1e-6)

        attn_probs = torch.nan_to_num(attn_probs, 0.0)
        attn_probs = torch.clamp(attn_probs, 0.0, 1.0)
        attn_probs = attn_probs / (torch.sum(attn_probs, dim=-1, keepdim=True) + 1e-6) # [batch, num_heads, 1, num_frame]

        attn_probs = self.attention_dropout(attn_probs)

        # Apply attention to Value
        attn_output = torch.matmul(attn_probs, value)  # [batch, num_heads, 1, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, 1, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, 1, self.hidden_size)  # [batch, 1, hidden_size]
        
        # Process vision information
        vision_pooled = self.vision_pool(normed_vision.mean(dim=1, keepdim=True))  # [batch, 1, hidden_size]

        # Multi-head attention output에 대한 projection & dropout
        fusion_emb = self.out_proj(attn_output)  # [batch, 1, hidden_size]
        fusion_emb = self.output_dropout(fusion_emb)  # [batch, 1, hidden_size]

        # Balanced fusion with learnable ratio
        alpha = torch.sigmoid(self.fusion_alpha)
        fusion_emb = fusion_emb + alpha * lang_out + (1 - alpha) * vision_pooled

        normed_fusion = self.ffn_norm(fusion_emb)  # [batch, 1, hidden_size]
        ffn_output = self.ffn(normed_fusion)  # [batch, 1, hidden_size]
        fusion_emb = ffn_output + fusion_emb  # [batch, 1, hidden_size]

        if not self.training:
            intermediate_tensors = [
                local_positions, global_positions, local_pos_emb, global_pos_emb,
                text_pos_emb, query, key, value, attn_scores, attn_probs,
                attn_output, vision_pooled, ffn_output
            ]
            for tensor in intermediate_tensors:
                del tensor
            torch.cuda.empty_cache()

        return fusion_emb.squeeze(1)  # [batch, hidden_size]


class ChapterHead(nn.Module):
    def __init__(self, lang_emb_size, vision_emb_size, hidden_size, num_heads, output_size):
        super().__init__()
        self.memory_manager = MemoryManager()
        self.hidden_size = hidden_size
        
        self.lang_proj_head = nn.Sequential(
            nn.LayerNorm(lang_emb_size),
            # nn.Dropout(0.1),
            nn.Linear(lang_emb_size, hidden_size),
            nn.GELU(),
            # nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        self.vision_proj_head = nn.Sequential(
            nn.LayerNorm(vision_emb_size),
            # nn.Dropout(0.1),
            nn.Linear(vision_emb_size, hidden_size),
            nn.GELU(),
            # nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.fusion_proj = None
        
        # self.cross_attention = CrossAttention(hidden_size, num_heads=num_heads)

        self._init_weights()

    def _init_weights(self):
        for module in [self.lang_proj_head, self.vision_proj_head]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(
                            layer.weight, 
                            mode='fan_out',
                            nonlinearity='relu'
                        )
                        nn.init.constant_(layer.bias, 0.01)
                    elif isinstance(layer, nn.LayerNorm):
                        nn.init.constant_(layer.weight, 1.0)
                        nn.init.constant_(layer.bias, 0.0)

        # Fusion norm은 별도로 초기화
        nn.init.constant_(self.fusion_norm.weight, 1.0)
        nn.init.constant_(self.fusion_norm.bias, 0.0)

        # Fusion projection이 있는 경우 초기화
        if self.fusion_proj is not None:
            nn.init.xavier_uniform_(
                self.fusion_proj.weight,
                gain=nn.init.calculate_gain('relu') * 0.1
            )
            nn.init.zeros_(self.fusion_proj.bias)


    def process_clip(self, lang_emb, vision_emb, clip_info):
        """        
        Args:
            lang_emb: [batch, hidden_size] 텍스트 임베딩
            vision_emb: [batch, num_frames, hidden_size] 비전 임베딩
            clip_info: {
                'clip_start_frame': [batch] 클립 시작 프레임 인덱스
                'total_frames': [batch] 비디오 전체 프레임 수
                'target_clip_idx': [batch] 타겟 클립 인덱스
                'total_num_clips': [batch] 전체 클립 수
            }
        Returns:
            fusion_emb: [batch, hidden_size] 융합된 임베딩
        """
        device = lang_emb.device

        if self.fusion_proj is None:
            seq_length = vision_emb.size(1) + 1  # num_frames + 1
            input_dim = seq_length * self.hidden_size
            self.fusion_proj = nn.Linear(input_dim, self.hidden_size).to(device)
        
        self.lang_proj_head = self.lang_proj_head.to(device)
        self.vision_proj_head = self.vision_proj_head.to(device)
        # self.cross_attention = self.cross_attention.to(device)
        self.fusion_proj = self.fusion_proj.to(device)

        lang_emb = lang_emb.to(device)
        vision_emb = vision_emb.to(device)
        clip_info = {key: value.to(device) for key, value in clip_info.items()}

        try:
            lang_proj = self.lang_proj_head(lang_emb)  # [batch, hidden_size]
            vision_proj = self.vision_proj_head(vision_emb)  # [batch, num_frames, hidden_size]
            concat_emb = torch.cat([vision_proj, lang_proj.unsqueeze(1)], dim=1) # [batch, num_frame + 1, hidden_size]

            batch_size = concat_emb.size(0)
            fusion_emb = concat_emb.view(batch_size, -1)
            fusion_emb = self.fusion_proj(fusion_emb)
            fusion_emb = self.fusion_norm(fusion_emb)
            fusion_emb = F.silu(fusion_emb)
            fusion_emb = F.dropout(fusion_emb, p=0.2, training=self.training)
            
            return fusion_emb
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.memory_manager.handle_oom()
                raise
            raise e


class TwoStream(nn.Module):
    def __init__(self, lang_model, vision_model, lang_embed_size, vision_embed_size,
                segment_size, hidden_size, window_size):
        super(TwoStream, self).__init__()
        self.lang_model = lang_model
        self.vision_model = vision_model
        self.segment_size = segment_size
        self.lang_embed_size = lang_embed_size
        self.vision_embed_size = vision_embed_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # 메모리 관리 초기화
        self.memory_manager = MemoryManager()
        self.cache_manager = self.memory_manager.get_cache_manager()

        logger.info("Initializing TwoStream model")

    def extract_language_features(self, text_ids, attention_mask):
        """Extract language features with caching"""
        cache_key = f"{text_ids.shape}_{hash(text_ids.cpu().numpy().tobytes())}"

        with torch.no_grad():
            return self.cache_manager.get_or_compute(
                owner=self,
                cache_name='lang_features',
                key=cache_key,
                compute_fn=lambda: self.lang_model(text_ids, attention_mask)["pooler_output"]
            )

    def extract_vision_features(self, frames):
        """Extract vision features with caching"""
        cache_key = f"{frames.shape}_{hash(frames.cpu().numpy().tobytes())}"

        with torch.no_grad():
            return self.cache_manager.get_or_compute(
                owner=self,
                cache_name='vision_features',
                key=cache_key,
                compute_fn=lambda: self.vision_model(frames)
            )

    def build_chapter_head(self, output_size):
        # Cross Attention head
        self.fusion_head = ChapterHead(
            self.lang_embed_size,
            self.vision_embed_size,
            self.hidden_size,
            num_heads=2,
            output_size=output_size
        )

        # Window self attention head
        window_config = type('Config', (), {
            'hidden_size': self.hidden_size,
            'num_attention_heads': 4,
            'attention_probs_dropout_prob': 0.1,
            'window_size': self.window_size
        })
        self.window_head = VideoChapterClassifier(window_config)


    def forward(self, img_clip, text_ids, attention_mask, clip_info):
        """
        Args:
            img_clip: [batch, num_clips, num_frames, C, H, W]
            text_ids: [batch, num_clips, seq_length]
            attention_mask: [batch, num_clips, seq_length]
            clip_info: Dictionary containing:
                - clip_start_frames: [batch, num_clips] 각 클립의 시작 프레임 인덱스
                - total_frames: [batch] 각 비디오의 총 프레임 수
                - target_clip_idx: [batch] 타겟 클립 인덱스
                - total_num_clips: [batch] 각 비디오의 총 클립 수
        """
        batch_size, num_clips, seq_length = text_ids.shape
        device = img_clip.device

        self.fusion_head = self.fusion_head.to(device)
        self.window_head = self.window_head.to(device)
        self.lang_model = self.lang_model.to(device)
        self.vision_model = self.vision_model.to(device)

         # Process language embeddings clip by clip
        clip_fusion_embs = []

        try:
            for i in range(num_clips):
                # Language processing
                clip_text_ids = text_ids[:, i, :].reshape(batch_size, -1)   # [batch_size, seq_length] 
                clip_attn_mask = attention_mask[:, i, :]

                current_clip_info = {
                    'clip_start_frame': clip_info['clip_start_frame'][:, i].to(device),
                    'total_frames': clip_info['total_frames'].to(device),
                    'target_clip_idx': clip_info['target_clip_idx'].to(device),
                    'total_num_clips': clip_info['total_num_clips'].to(device)
                }
                
                # feature extract
                lang_emb = self.extract_language_features(clip_text_ids, clip_attn_mask).to(device)  # [batch, 768] # [batch_size, hidden_size]

                # Vision processing
                clip_frames = img_clip[:, i]
                frames = rearrange(clip_frames, 'b nf c h w -> (b nf) c h w')
                frame_emb = self.extract_vision_features(frames).to(device)
                frame_emb = frame_emb.view(batch_size, self.segment_size, -1).float() # [batch, 16, 2048]
                # print(f'Original vision emb size: {frame_emb.shape}')
                # print(f'Original lang emb size: {lang_emb.shape}')
                clip_fusion = self.fusion_head.process_clip(lang_emb, frame_emb, current_clip_info)
                clip_fusion_embs.append(clip_fusion)
                
                # del lang_emb, clip_frames, frames, frame_emb

            # Stack all embeddings
            all_fusion_embs = torch.stack(clip_fusion_embs, dim=1).to(device)
            binary_logits, binary_prob = self.window_head(all_fusion_embs, clip_info)

            # del all_fusion_embs, clip_fusion_embs
            # self.memory_manager.cleanup()
            
            if not self.training:
                del clip_fusion_embs
                del all_fusion_embs
                torch.cuda.empty_cache()
            
            return binary_logits, binary_prob
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.memory_manager.handle_oom()
            raise e
    
    def configure_optimizers(self, train_config):
        pretrained_params = []
        attn_params = []

        for name, param in self.named_parameters():
            if 'lang_model.base_model' in name or 'vision_model.base_model' in name:
                pretrained_params.append(param)
            else:
                attn_params.append(param)

        optim_groups = [
            {
                "params":pretrained_params,
                'lr':train_config.learning_rate*0.1,
                'weight_decay':train_config.weight_decay*0.1
            },
            {
                "params":attn_params,
                'lr':train_config.learning_rate,
                'weight_decay':train_config.weight_decay
            }
        ]

        optimizer = torch.optim.AdamW(optim_groups, betas=train_config.betas, weight_decay=0.0)
        return optimizer

    # def configure_optimizers(self, train_config):
    #     """
    #     Configure optimizer with parameter groups for different components
    #     """
    #     # Pre-trained models parameters (fine-tuning with smaller lr)
    #     bert_params = [
    #         p for n, p in self.named_parameters() 
    #         if 'lang_model' in n and p.requires_grad
    #     ]
    #     resnet_params = [
    #         p for n, p in self.named_parameters() 
    #         if 'vision_model' in n and p.requires_grad
    #     ]

    #     # Fusion head parameters (projection layers and fusion)
    #     fusion_params = [
    #         p for n, p in self.named_parameters() 
    #         if 'fusion_head' in n and p.requires_grad
    #     ]

    #     # Window attention parameters
    #     window_params = [
    #         p for n, p in self.named_parameters() 
    #         if 'window_head.window_block' in n and p.requires_grad
    #     ]

    #     # Classifier parameters
    #     classifier_params = [
    #         p for n, p in self.named_parameters() 
    #         if 'window_head.classifier' in n and p.requires_grad
    #     ]

    #     # Configure parameter groups with different learning rates and weight decay
    #     optim_groups = [
    #         {
    #             "params": bert_params,
    #             "lr": train_config.learning_rate*0.1,  # 매우 작은 lr
    #             "weight_decay": train_config.weight_decay * 0.1,
    #             "name": "bert"
    #         },
    #         {
    #             "params": resnet_params,
    #             "lr": train_config.learning_rate*0.1,  # 매우 작은 lr
    #             "weight_decay": train_config.weight_decay * 0.1,
    #             "name": "resnet"
    #         },
    #         {
    #             "params": fusion_params,
    #             "lr": train_config.learning_rate*3,
    #             "weight_decay": train_config.weight_decay*0.2,
    #             "name": "fusion"  # for logging purposes
    #         },
    #         {
    #             "params": window_params,
    #             "lr": train_config.learning_rate*2,
    #             "weight_decay": train_config.weight_decay*0.5,
    #             "name": "window_attention"
    #         },
    #         {
    #             "params": classifier_params,
    #             "lr": train_config.learning_rate,
    #             "weight_decay": train_config.weight_decay,
    #             "name": "classifier"
    #         }
    #     ]

    #     # Filter out empty groups
    #     optim_groups = [g for g in optim_groups if len(list(g["params"])) > 0]

    #     # Create optimizer
    #     optimizer = torch.optim.AdamW(
    #         optim_groups,
    #         betas=(0.9, 0.999),
    #         eps=1e-8,
    #         weight_decay=0.0  # weight decay is handled per group
    #     )

    #     return optimizer