import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from memory_cache_utils import MemoryManager

class VideoChapterWindowAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, window_size, dropout=0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = window_size

        self.memory_manager = MemoryManager()
        
        # Linear Layer to transform Query, Key, Value
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropouts
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Position and progress embedding
        self.position_encoding = nn.Sequential(
            nn.Linear(2, hidden_size//2),  # [normalized_pos, progress]
            nn.LayerNorm(hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

        # Learnable window position bias
        self.window_pos_bias = nn.Parameter(torch.zeros(1, num_attention_heads, 1, window_size))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.attention_head_size)

        # Initialize attention weights
        for module in [self.query, self.key, self.value, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=scale)
            nn.init.zeros_(module.bias)

        # Initialize window position bias
        nn.init.normal_(self.window_pos_bias, mean=0.0, std=0.02)

        # Initialize position encoding weights
        for layer in self.position_encoding:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def get_clip_positions(self, clip_indices, total_clips):
        """
        Args:
            clip_indices: window 내 클립들의 전역 인덱스 [window_size]
            total_clips: 비디오의 총 클립 수 (scalar)
        """
        window_size = len(clip_indices)
        middle_idx = window_size // 2

        # 1. Window 내 상대적 위치 (-1 ~ 1)
        local_positions = torch.arange(window_size, device=clip_indices.device) - middle_idx
        local_positions = local_positions.float() / (middle_idx + 1e-6)

        # 2. 전체 비디오에서의 절대적 위치
        clip_indices = torch.clamp(clip_indices.float(), 0, total_clips-1)
        global_positions = torch.log(clip_indices + 1) / torch.log(total_clips + 1)

        # 두 위치 정보 결합
        position_info = torch.stack([local_positions, global_positions], dim=-1)
        return position_info

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_size)

    def forward(self, hidden_states, clip_info):
        """
        Args:
            hidden_states: [batch_size, window_size, hidden_size]
            clip_info: {
                'clip_start_frame': [batch, num_clips],
                'total_frames': [batch],
                'target_clip_idx': [batch],
                'total_num_clips': [batch]
            }
        """
        try:
            batch_size, seq_length, _ = hidden_states.shape
            middle_idx = seq_length // 2  # target clip index
            device = hidden_states.device

            # Calculate position information
            position_info = []
            for i in range(batch_size):
                target_idx = clip_info['target_clip_idx'][i]
                total_clips = clip_info['total_num_clips'][i]
                
                start = target_idx - middle_idx
                window_indices = torch.arange(start, start + seq_length, device=device)
                window_indices = torch.clamp(window_indices, 0, total_clips - 1)
                
                pos_info = self.get_clip_positions(window_indices, total_clips)
                position_info.append(pos_info)
                
            position_info = torch.stack(position_info, dim=0)  # [batch_size, seq_length, 2]
            
            # Generate position embeddings
            position_embeddings = self.position_encoding(position_info)  # [batch_size, seq_length, hidden_size]

            # Add position embeddings to input
            hidden_states = hidden_states + position_embeddings

            # Generate Q, K, V layer
            target_hidden = hidden_states[:, middle_idx:middle_idx+1, :]  # [batch, 1, hidden]
            query_layer = self.transpose_for_scores(self.query(target_hidden))  # [batch, heads, 1, head_size]
            key_layer = self.transpose_for_scores(self.key(hidden_states))    # [batch, heads, num_clips, head_size]
            value_layer = self.transpose_for_scores(self.value(hidden_states))  # [batch, heads, num_clips, head_size]
            
            # Attention 계산
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [batch, heads, 1, num_clips]
            attention_scale = math.sqrt(self.attention_head_size) + 1e-6
            attention_scores = attention_scores / attention_scale

            self.window_pos_bias.data = torch.clamp(self.window_pos_bias.data, -10, 10)
            attention_scores = attention_scores + self.window_pos_bias

            attention_scores = torch.clamp(attention_scores, -10, 10)
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.attention_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            context_layer = context_layer.view(batch_size, 1, self.all_head_size)

            # Output projection with dropout
            attention_output = self.out_proj(context_layer) # [batch, 1, hidden]
            attention_output = self.output_dropout(attention_output)

            # 추론 시에만 중간 결과 정리
            if not self.training:
                intermediate_tensors = [
                    position_info, position_embeddings,
                    query_layer, key_layer, value_layer,
                    attention_scores, attention_probs, context_layer
                ]
                for tensor in intermediate_tensors:
                    del tensor
                torch.cuda.empty_cache()
            
            return attention_output
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.memory_manager.cleanup(force=True)
                raise RuntimeError("Memory error in attention computation. Please try with smaller batch size.")
            raise e


class VideoChapterBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, window_size, dropout=0.1):
        super().__init__()
        self.memory_manager = MemoryManager()
        
        self.attention_layernorm = nn.LayerNorm(hidden_size)
        self.ffn_layernorm = nn.LayerNorm(hidden_size)

        self.attention = VideoChapterWindowAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.2)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, hidden_states, clip_info): # [batch, num_clips, hidden_size]
        try:
            # Attention layer with residual connection
            middle_idx = hidden_states.size(1) // 2
            residual = hidden_states[:, middle_idx:middle_idx+1, :] # [batch, 1, hidden_size]

            # Feed Forward layer with residual connection
            normed_states = self.attention_layernorm(hidden_states)
            attention_output = self.attention(normed_states, clip_info) # [batch, 1, hidden_size]
            attention_output = attention_output + residual

            normed_attention = self.ffn_layernorm(attention_output)
            ffn_output = self.ffn(normed_attention)
            output = ffn_output + attention_output # [batch, 1, hidden_size]
            
            if not self.training:
                del normed_states, attention_output, normed_attention, ffn_output
                torch.cuda.empty_cache()

            return output
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.memory_manager.cleanup(force=True)
            raise e


class VideoChapterClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory_manager = MemoryManager()

        self.window_block = VideoChapterBlock(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            window_size=config.window_size,
            dropout=config.attention_probs_dropout_prob
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size//2, 2)
        )

    def _init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, fusion_emb, clip_info):
        device = fusion_emb.device
        if not hasattr(self, '_modules_moved'):
            self.window_block = self.window_block.to(device)
            self.classifier = self.classifier.to(device)
            self._modules_moved = True
        
        try:
            attention_output = self.window_block(fusion_emb, clip_info)
            logits = self.classifier(attention_output.squeeze(1))
            probs = F.softmax(logits, dim=-1)

            if not self.training:
                del attention_output
                torch.cuda.empty_cache()
            
            return logits, probs

        except RuntimeError as e:
            if "out of memory" in str(e):
                self.memory_manager.cleanup(force=True)
            raise e