import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VideoChapterWindowAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value 변환을 위한 선형 레이어
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        # 입력 텐서를 attention 계산을 위한 형태로 변환
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_size)

    def _compute_attention(self, query, key, value):
        # Target clip의 query와 모든 clip의 key, value로 attention 계산
        batch_size = query.size(0)
        
        # Attention 점수 계산
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) # [batch, heads, 1, num_clips]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Attention weights 계산
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Context 계산
        context_layer = torch.matmul(attention_probs, value)  # [batch, heads, 1, head_size]
        
        # Multi-head 결과 병합
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, 1, self.all_head_size)
        
        return context_layer

    def forward(self, hidden_states):
        # target clip이 모든 clip과 attention하는 프로세스
        batch_size, num_clips, _ = hidden_states.size()
        middle_idx = num_clips // 2  # target clip index
        
        # Target clip에 대한 query 생성
        target_hidden = hidden_states[:, middle_idx:middle_idx+1, :]  # [batch, 1, hidden]
        query_layer = self.transpose_for_scores(self.query(target_hidden))  # [batch, heads, 1, head_size]
        
        # 모든 clip에 대한 key, value 생성
        key_layer = self.transpose_for_scores(self.key(hidden_states))    # [batch, heads, num_clips, head_size]
        value_layer = self.transpose_for_scores(self.value(hidden_states))  # [batch, heads, num_clips, head_size]
        
        # Attention 계산
        context_layer = self._compute_attention(query_layer, key_layer, value_layer)
        
        # Output projection
        output = self.out_proj(context_layer)  # [batch, 1, hidden]
        
        return output


class VideoChapterBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        self.attention = VideoChapterWindowAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, hidden_states):
        # Attention layer with residual connection
        middle_idx = hidden_states.size(1) // 2
        attention_output = self.attention(hidden_states)
        attention_output = self.layer_norm1(attention_output + hidden_states[:, middle_idx:middle_idx+1, :])
        
        # Feed Forward layer with residual connection
        ffn_output = self.ffn(attention_output)
        output = self.layer_norm2(ffn_output + attention_output)
        
        return output


class VideoChapterClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window_block = VideoChapterBlock(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size//2),
            nn.GELU(),
            nn.Dropout(config.attention_probs_dropout_prob),
            nn.Linear(config.hidden_size//2, 2)
        )

    def forward(self, fusion_emb):
        # Target attention block
        attention_output = self.window_block(fusion_emb)
        
        # Classification
        logits = self.classifier(attention_output.squeeze(1))
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs