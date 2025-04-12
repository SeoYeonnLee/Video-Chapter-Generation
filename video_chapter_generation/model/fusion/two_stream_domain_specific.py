import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math
from memory_cache_utils import MemoryManager


class WindowSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size, dropout=0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of attention "
                f"heads {num_heads}."
            )
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size*2, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size*2, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size*2, hidden_size)
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Position bias
        self.window_pos_bias = nn.Parameter(torch.zeros(1, num_heads, 2*window_size+1, 2*window_size+1))
        
        # Position embeddings
        self.position_encoding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.head_dim)

        nn.init.xavier_uniform_(self.query_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=scale)
        
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        
        for module in self.out_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                
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

    def forward(self, hidden_states, clip_info=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        # Add position embeddings
        position_info = self.get_relative_positions(seq_length)
        position_embeddings = self.position_encoding(position_info)
        hidden_states = hidden_states + position_embeddings
        
        # Layer normalization
        hidden_states = self.norm(hidden_states)
        
        # Project query, key, value
        query = self.query_proj(hidden_states)
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Add position bias
        if seq_length <= self.window_pos_bias.size(-1):
            attention_scores = attention_scores + self.window_pos_bias[:, :, :seq_length, :seq_length]
        
        # Attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        output = self.output_dropout(output)
        
        return output


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of attention "
                f"heads {num_heads}."
            )
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        # self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size*2, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size*2, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size*2, hidden_size)
        )

        self.vision_norm = nn.LayerNorm(hidden_size)
        self.lang_norm = nn.LayerNorm(hidden_size)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.head_dim)

        nn.init.xavier_uniform_(self.query_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=scale)
        
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)

        for module in self.out_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, lang_emb, vision_emb):
        """
        lang_emb: [batch, hidden_size]
        vision_emb: [batch, hidden_size]
        """
        batch_size = lang_emb.shape[0]
        
        # Normalize inputs
        lang_emb = self.lang_norm(lang_emb)
        vision_emb = self.vision_norm(vision_emb)
        
        # Project query (language) and key/value (vision)
        query = self.query_proj(lang_emb).unsqueeze(1)  # [batch, 1, hidden_size]
        key = self.key_proj(vision_emb).unsqueeze(1)    # [batch, 1, hidden_size]
        value = self.value_proj(vision_emb).unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Reshape for multi-head attention
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, 1, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        output = self.output_dropout(output)
        
        return output.squeeze(1)


class ChapterHead(nn.Module):
    def __init__(self, lang_emb_size, vision_emb_size, segment_size, hidden_size, window_size, output_size):
        super(ChapterHead, self).__init__()
        self.lang_emb_size = lang_emb_size
        self.vision_emb_size = vision_emb_size
        self.segment_size = segment_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_clips = 2 * window_size + 1
        
        # Projection for language embeddings
        self.lang_proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lang_emb_size, lang_emb_size//2), 
                nn.LayerNorm(lang_emb_size//2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(lang_emb_size//2, hidden_size), 
            ) for _ in range(self.num_clips)
        ])

        # Projection for vision embeddings
        self.vision_proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vision_emb_size, 8 * hidden_size), 
                nn.LayerNorm(8 * hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(8 * hidden_size, 4 * hidden_size),
                nn.LayerNorm(4 * hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1), 
                nn.Linear(4 * hidden_size, hidden_size)
            ) for _ in range(self.num_clips)
        ])
        
        # Window self-attention for language and vision
        self.lang_window_attn = WindowSelfAttention(
            hidden_size=hidden_size,
            num_heads=16,
            window_size=window_size,
            dropout=0.1
        )
        
        self.vision_window_attn = WindowSelfAttention(
            hidden_size=hidden_size,
            num_heads=16,
            window_size=window_size,
            dropout=0.1
        )
        
        # Cross-attention between language and vision
        self.cross_attn = CrossAttention(
            hidden_size=hidden_size,
            num_heads=16,
            dropout=0.1
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.LayerNorm(hidden_size//4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size//4, output_size)
        )

    def forward(self, lang_embs, vision_embs, clip_info):
        """
        lang_embs: [batch, num_clips, lang_emb_size]
        vision_embs: [batch, num_clips, segment_size, vision_emb_size]
        """
        batch_size, num_clips = lang_embs.shape[:2]
        
        # Project language and vision embeddings
        projected_lang_embs = []
        projected_vision_embs = []
        
        for i in range(num_clips):
            # Project language embeddings
            lang_emb = lang_embs[:, i]  # [batch, lang_emb_size]
            projected_lang = self.lang_proj_heads[i](lang_emb)  # [batch, hidden_size]
            projected_lang = F.relu(projected_lang)
            projected_lang_embs.append(projected_lang)
            
            # Project vision embeddings
            vision_emb = vision_embs[:, i]  # [batch, segment_size, vision_emb_size]
            vision_emb = vision_emb.reshape(-1, self.vision_emb_size)
            projected_vision = self.vision_proj_heads[i](vision_emb)
            projected_vision = projected_vision.view(batch_size, self.segment_size, self.hidden_size)
            projected_vision = F.relu(projected_vision)
            
            # Mean pool along segment dimension
            pooled_vision = projected_vision.mean(dim=1)  # [batch, hidden_size]
            projected_vision_embs.append(pooled_vision)
        
        # Stack projected embeddings
        lang_tensor = torch.stack(projected_lang_embs, dim=1)  # [batch, num_clips, hidden_size]
        vision_tensor = torch.stack(projected_vision_embs, dim=1)  # [batch, num_clips, hidden_size]
        
        # Apply window self-attention to language and vision
        lang_attended = self.lang_window_attn(lang_tensor, clip_info)  # [batch, num_clips, hidden_size]
        vision_attended = self.vision_window_attn(vision_tensor, clip_info)  # [batch, num_clips, hidden_size]
        
        # Extract center clip (target)
        center_idx = num_clips // 2
        lang_center = lang_attended[:, center_idx]  # [batch, hidden_size]
        vision_center = vision_attended[:, center_idx]  # [batch, hidden_size]
        
        # Apply cross-attention between language and vision
        fusion_emb = self.cross_attn(lang_center, vision_center)  # [batch, hidden_size]
        
        # Classify
        logits = self.classifier(fusion_emb)  # [batch, output_size]
        
        return logits


class TwoStream(nn.Module):
    def __init__(self, lang_model, vision_model, lang_embed_size, vision_embed_size, segment_size, hidden_size, window_size):
        super(TwoStream, self).__init__()
        self.lang_model = lang_model
        self.vision_model = vision_model
        self.segment_size = segment_size
        self.lang_embed_size = lang_embed_size
        self.vision_embed_size = vision_embed_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        self.memory_manager = MemoryManager()
        self.cache_manager = self.memory_manager.get_cache_manager()

    def build_chapter_head(self, output_size, head_type=None):
        """
        build a new head for video chapter prediction
        head_type: Not used in this implementation (kept for compatibility)
        """
        self.fusion_head = ChapterHead(
            self.lang_embed_size,
            self.vision_embed_size,
            self.segment_size,
            self.hidden_size,
            self.window_size,
            output_size
        )

    def configure_optimizers(self, train_config):
        """
        Separates parameters into weight decay and no weight decay groups
        """
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif "LayerNorm" in fpn:
                    no_decay.add(fpn)
                elif "bn" in fpn:
                    no_decay.add(fpn)
                elif "emb" in fpn:
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, img_clips, text_ids, attention_masks, clip_info):
        batch_size, num_clips, seq_length = text_ids.shape
        device = img_clips.device
        
        # Ensure models are on the correct device
        self.lang_model = self.lang_model.to(device)
        self.vision_model = self.vision_model.to(device)
        
        # Extract embeddings for all clips
        all_lang_embs = []
        all_vision_embs = []
        for i in range(num_clips):
            clip_text_ids = text_ids[:, i, :].reshape(batch_size, -1)
            clip_attn_mask = attention_masks[:, i, :]
            
            lang_model_inputs = {
                "input_ids": clip_text_ids,
                "attention_mask": clip_attn_mask
            }
            lang_model_output = self.lang_model(**lang_model_inputs)
            lang_emb = lang_model_output.pooler_output
            all_lang_embs.append(lang_emb)
        
            img_clip = img_clips[:, i]
            img_clip = rearrange(img_clip, 'b t c h w -> (b t) c h w').contiguous()
            vision_emb = self.vision_model(img_clip)
            vision_emb = vision_emb.view(batch_size, self.segment_size, -1)
            all_vision_embs.append(vision_emb)
        
        # Stack embeddings
        stacked_lang_embs = torch.stack(all_lang_embs, dim=1)  # [batch, num_clips, lang_embed_size]
        stacked_vision_embs = torch.stack(all_vision_embs, dim=1)  # [batch, num_clips, segment_size, vision_embed_size]
        
        # Process through fusion head
        binary_logits = self.fusion_head(stacked_lang_embs, stacked_vision_embs, clip_info)
        binary_prob = F.softmax(binary_logits, dim=1)
        
        return binary_logits, binary_prob