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

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # self.out_proj = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size * 2),
        #     nn.LayerNorm(hidden_size * 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),

        #     nn.Linear(hidden_size * 2, hidden_size)
        # )
        
        self.attention_dropout = nn.Dropout(0.2)

        self.position_encoding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

        self.window_pos_bias = nn.Parameter(torch.zeros(1, num_attention_heads, 1, 2*window_size+1))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.out_proj.bias)
        
        nn.init.normal_(self.window_pos_bias, mean=0.0, std=0.02)
        
        for m in self.position_encoding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_relative_positions(self, seq_length):
        middle_idx = seq_length // 2
        positions = torch.arange(seq_length, device=self.window_pos_bias.device)
        positions = (positions - middle_idx).float() / (middle_idx + 1e-6)
        return positions.unsqueeze(-1)  # [seq_length, 1]
        

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
        batch_size, seq_length, _ = hidden_states.shape
        middle_idx = seq_length // 2  # target clip index

        position_info = self.get_relative_positions(seq_length)
        position_embeddings = self.position_encoding(position_info)

        # Add position embeddings to input
        hidden_states = hidden_states + position_embeddings

        # Generate Q, K, V layer
        target_hidden = hidden_states[:, middle_idx:middle_idx+1, :]
        query_layer = self.transpose_for_scores(self.query(target_hidden)) 
        key_layer = self.transpose_for_scores(self.key(hidden_states))  
        value_layer = self.transpose_for_scores(self.value(hidden_states)) 
        
        # Attention 계산
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + self.window_pos_bias[:, :, :, :seq_length]

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, 1, self.all_head_size)

        attention_output = self.out_proj(context_layer) # [batch, 1, hidden]

        return attention_output


class VideoChapterBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, window_size, dropout=0.1):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)

        self.attention = VideoChapterWindowAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        self.ffn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.15)
        )

        self._init_weights()

    def _init_weights(self):
        
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states, clip_info): # [batch, num_clips, hidden_size]
        
        middle_idx = hidden_states.size(1) // 2
        residual = hidden_states[:, middle_idx:middle_idx+1, :] # [batch, 1, hidden_size]

        normed_states = self.attention_norm(hidden_states)
        attention_output = self.attention(normed_states, clip_info)
        attention_output = attention_output + residual

        residual = attention_output
        normed_attention = self.ffn_norm(attention_output)
        ffn_output = self.ffn(normed_attention)
        output = ffn_output + residual

        return output


class VideoChapterClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.window_block = VideoChapterBlock(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            window_size=config.window_size,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size//2, 2)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, fusion_emb, clip_info):
        
        attention_output = self.window_block(fusion_emb, clip_info)
        logits = self.classifier(attention_output.squeeze(1))
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs