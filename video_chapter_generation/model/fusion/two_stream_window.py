import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math
from memory_cache_utils import MemoryManager
# from model.fusion.window_self_attention import VideoChapterClassifier
from model.fusion.stacked_window_self_attention import StackedVideoChapterAttention


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
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
        self.lang_norm = nn.LayerNorm(hidden_size)
        self.vision_norm = nn.LayerNorm(hidden_size)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        

        self.frame_pos_encoding = nn.Linear(1, hidden_size)
        
        self._init_weights()

    def _init_weights(self):

        scale = 1.0 / math.sqrt(self.head_dim)
        for proj in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=scale)
            nn.init.zeros_(proj.bias)

        nn.init.xavier_uniform_(self.frame_pos_encoding.weight)
        nn.init.zeros_(self.frame_pos_encoding.bias)


    def get_relative_positions(self, num_frames):
        positions = torch.arange(num_frames, device=self.query_proj.weight.device).float()
        positions = positions / (num_frames - 1)  # Normalize to [0, 1]
        return positions.unsqueeze(-1) 

    def forward(self, lang_emb, vision_emb):
        batch_size, num_frames, _ = vision_emb.shape

        lang_emb = self.lang_norm(lang_emb)
        vision_emb = self.vision_norm(vision_emb)
        
        position_info = self.get_relative_positions(num_frames)
        position_emb = self.frame_pos_encoding(position_info)
        vision_emb = vision_emb + position_emb
        
        query = self.query_proj(lang_emb)
        key = self.key_proj(vision_emb) 
        value = self.value_proj(vision_emb)

        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        key = key.view(batch_size, num_frames, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, num_frames, head_dim]
        value = value.view(batch_size, num_frames, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, num_frames, head_dim]

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # [batch, num_heads, 1, num_frames]
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        context = torch.matmul(attention_weights, value)  # [batch, num_heads, 1, head_dim]
        
        context = context.transpose(1, 2).contiguous()  # [batch, 1, num_heads, head_dim]
        context = context.view(batch_size, 1, self.hidden_size)  # [batch, 1, hidden_size]

        output = self.out_proj(context)
        output = self.output_dropout(output)
        
        return output.squeeze(1) 

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, n_embd, n_head, output_size, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(n_embd, output_size)
        
 
    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v     # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # re-assemble all head outputs side by side

        # output projection
        y = y[:, 0, :].squeeze(1)   # use the first token as fused embedding
        out = self.proj(y)
        return out


class ChapterHead(nn.Module):
    def __init__(self, lang_emb_size, vision_emb_size, segment_size, hidden_size, window_size, output_size, head_type="mlp"):
        super(ChapterHead, self).__init__()
        self.lang_emb_size = lang_emb_size
        self.vision_emb_size = vision_emb_size
        self.segment_size = segment_size
        self.hidden_size = hidden_size
        self.head_type = head_type
        self.window_size = window_size
        self.num_clips = 2 * window_size + 1

        # self.lang_proj_head = nn.Linear(lang_emb_size, hidden_size, bias=False)
        self.lang_proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lang_emb_size, lang_emb_size//2), 
                nn.LayerNorm(lang_emb_size//2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(lang_emb_size//2, hidden_size), 
            ) for _ in range(self.num_clips)
        ])

        # self.vision_proj_head = nn.Linear(vision_emb_size, hidden_size, bias=False)
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

        if head_type == "mlp":
            print(f'head type: mlp')
            # self.head = nn.Linear((segment_size + 1) * hidden_size, hidden_size, bias=True)
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear((segment_size + 1) * hidden_size, 8 * hidden_size), 
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
        elif head_type == "bilinear":
            print(f'head type: bilinear')
            self.bilinear_layers = nn.ModuleList([
                nn.Bilinear(hidden_size, hidden_size * segment_size, hidden_size * 2)
                for _ in range(self.num_clips)
            ])
            
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden_size * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 2, hidden_size * 1),
                    nn.LayerNorm(hidden_size * 1),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 1, hidden_size),
                ) for _ in range(self.num_clips)
            ])

        elif head_type == "multiplication":
            print(f'head type: multiplication')
            self.lang_expand_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 8),
                    nn.LayerNorm(hidden_size * 8),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 8, hidden_size * segment_size),
                    nn.LayerNorm(hidden_size * segment_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ) for _ in range(self.num_clips)
            ])
            
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size * segment_size, hidden_size * 8),
                    nn.LayerNorm(hidden_size * 8),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 8, hidden_size * 4),
                    nn.LayerNorm(hidden_size * 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 4, hidden_size)
                ) for _ in range(self.num_clips)
            ])
            
        elif head_type == "self_attn":
            self.head = SelfAttention(hidden_size, 4, hidden_size)

        elif self.head_type == "cross_attn":
            print(f'head type: cross attention')
            self.head = CrossAttention(hidden_size, num_heads=16)
            self.output_proj = nn.Linear(hidden_size, output_size)

        else:
            raise RuntimeError(f"Unknown head_type {head_type}")


    def forward(self, lang_emb, vision_emb, window_idx):
        """
        lang_emb: [batch, lang_emb_size]
        vision_emb: [batch, segment_size, vision_emb]
        """

        batch_size = lang_emb.shape[0]

        # lang_out = self.lang_proj_head(lang_emb).unsqueeze(1)        # batch, 1, hidden_size
        lang_out = self.lang_proj_heads[window_idx](lang_emb)
        lang_out = F.relu(lang_out)

        vision_emb = vision_emb.view(-1, self.vision_emb_size)
        vision_out = self.vision_proj_heads[window_idx](vision_emb).view(batch_size, self.segment_size, self.hidden_size)  # batch, segment, hidden_size
        vision_out = F.relu(vision_out)

        if self.head_type == "mlp":
            fusion_emb = torch.cat([vision_out, lang_out.unsqueeze(1)], dim=1)
            fusion_emb = fusion_emb.view(batch_size, -1)
            fusion_emb = self.head[window_idx](fusion_emb)

        elif self.head_type == "bilinear":
            vision_flat = vision_out.view(batch_size, -1)  # [batch, segment_size * hidden_size]
            fusion_emb = self.bilinear_layers[window_idx](lang_out, vision_flat)
            fusion_emb = self.head[window_idx](fusion_emb)  # [batch, hidden_size * 4]

        elif self.head_type == "multiplication":
            expanded_lang = self.lang_expand_layers[window_idx](lang_out)
            expanded_lang = expanded_lang.view(batch_size, self.segment_size, self.hidden_size)
            mul_fusion = vision_out * expanded_lang
            mul_fusion_flat = mul_fusion.view(batch_size, -1)
            fusion_emb = self.head[window_idx](mul_fusion_flat)
            
        elif self.head_type == "self_attn":
            fusion_emb = torch.cat([vision_out, lang_out.unsqueeze(1)], dim=1)
            fusion_emb = self.head(fusion_emb) # 검수 필요

        elif self.head_type == "cross_attn":
            fusion_emb = self.head(lang_out, vision_out)  # [batch, hidden_size]
            out = self.output_proj(fusion_emb)

        return fusion_emb
        # return out

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

        self.window_mlp = nn.Sequential(
            nn.Linear(hidden_size * (2 * window_size+ 1), hidden_size), 
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

            nn.Linear(hidden_size//4, hidden_size//8), 
            nn.LayerNorm(hidden_size//8),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size//8, hidden_size//16), 
            nn.LayerNorm(hidden_size//16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size//16, 2)
        )

    def build_chapter_head(self, output_size, head_type="mlp"):
        """
        build a new head for video chapter prediction
        head_type: mlp or attn
        """

        self.fusion_head = ChapterHead(
            self.lang_embed_size,
            self.vision_embed_size,
            self.segment_size,
            self.hidden_size,
            self.window_size,
            output_size,
            head_type
        )
        
        window_config = type('Config', (), {
            'hidden_size': self.hidden_size,
            'num_attention_heads': 16,
            'attention_probs_dropout_prob': 0.1,
            'window_size': self.window_size
        })
        self.window_attn = StackedVideoChapterAttention(window_config)

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif "LayerNorm" in fpn:
                    no_decay.add(fpn)
                elif "bn" in fpn:
                    no_decay.add(fpn)
                elif "emb" in fpn:
                    no_decay.add(fpn)
                else:
                    # weights of whitelist modules will be weight decayed
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
        self.lang_model = self.lang_model.to(device)
        self.vision_model = self.vision_model.to(device)
        self.window_attn = self.window_attn.to(device)

        clip_fusion_embs = []

        for i in range(num_clips):
            window_idx = i
            clip_text_ids = text_ids[:, i, :].reshape(batch_size, -1)   # [batch_size, seq_length] 
            clip_attn_mask = attention_masks[:, i, :]

            current_clip_info = {
                'clip_start_frame': clip_info['clip_start_frame'][:, i].to(device),
                'total_frames': clip_info['total_frames'].to(device),
                'target_clip_idx': clip_info['target_clip_idx'].to(device),
                'total_num_clips': clip_info['total_num_clips'].to(device)
                }

            # language
            lang_model_inputs = {
                "input_ids": clip_text_ids,
                "attention_mask": clip_attn_mask
            }
            lang_model_output = self.lang_model(**lang_model_inputs)
            lang_emb = lang_model_output.pooler_output

            # vision
            img_clip = img_clips[:, i]
            img_clip = rearrange(img_clip, 'b t c h w -> (b t) c h w').contiguous()

            vision_emb = self.vision_model(img_clip)

            vision_emb = vision_emb.view(batch_size, self.segment_size, -1) # [batch, 16, 2048]


            clip_fusion = self.fusion_head(lang_emb, vision_emb, window_idx) # [batch, hidden_size]
            clip_fusion_embs.append(clip_fusion)

        all_fusion_embs = torch.stack(clip_fusion_embs, dim=1).to(device) # [batch, num_clips, hidden_size]
        # MLP
        # all_fusion_embs = all_fusion_embs.view(batch_size, -1)
        # binary_logits = self.window_mlp(all_fusion_embs)
        # binary_prob = F.softmax(binary_logits, dim=1)

        # Self Attention
        binary_logits, binary_prob = self.window_attn(all_fusion_embs, clip_info)
        
        return binary_logits, binary_prob