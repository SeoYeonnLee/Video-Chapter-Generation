import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math
import gc
from model.fusion.window_self_attention import VideoChapterWindowAttention, VideoChapterBlock, VideoChapterClassifier

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lang_out, vision_out):
        """
        lang_out: [batch, 1, hidden_size]  # 한 clip의 language embedding
        vision_out: [batch, num_frame, hidden_size]  # 한 clip의 vision embedding
        """
        batch_size, num_frame = vision_out.shape[:2]
        
        query = self.query_proj(lang_out)  # [batch, 1, hidden_size]
        key = self.key_proj(vision_out)    # [batch, num_frame, hidden_size]
        value = self.value_proj(vision_out) # [batch, num_frame, hidden_size]

        # Reshape for multi-head attention
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, num_frame, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, num_frame, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to Value
        attn_output = torch.matmul(attn_probs, value)  # [batch, num_heads, 1, head_dim]
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, 1, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, 1, self.hidden_size)  # [batch, 1, hidden_size]
        fusion_emb = self.out_proj(attn_output)  # [batch, 1, hidden_size]

        return fusion_emb.squeeze(1)  # [batch, hidden_size]

class ChapterHead(nn.Module):
    def __init__(self, lang_emb_size, vision_emb_size, hidden_size, num_heads, output_size):
        super().__init__()
        self.lang_proj_head = nn.Linear(lang_emb_size, hidden_size)
        self.vision_proj_head = nn.Linear(vision_emb_size, hidden_size)
        self.cross_attention = CrossAttention(hidden_size, num_heads=8)
        self.final_mlp = nn.Linear(hidden_size, output_size)

    def process_clip(self, lang_emb, vision_emb):
        """각 clip에 대한 처리"""
        # Projection
        lang_proj = F.relu(self.lang_proj_head(lang_emb))  # [batch, hidden_size]
        vision_proj = F.relu(self.vision_proj_head(vision_emb))  # [batch, num_frames, hidden_size]
        
        # Cross attention
        fusion_emb = self.cross_attention(
            lang_proj.unsqueeze(1),  # [batch, 1, hidden_size]
            vision_proj              # [batch, num_frames, hidden_size]
        )  # [batch, hidden_size]
        
        return fusion_emb


class TwoStream(nn.Module):
    def __init__(self, lang_model, vision_model, lang_embed_size, vision_embed_size, segment_size, hidden_size):
        super(TwoStream, self).__init__()
        self.lang_model = lang_model
        self.vision_model = vision_model
        self.segment_size = segment_size
        self.lang_embed_size = lang_embed_size
        self.vision_embed_size = vision_embed_size
        self.hidden_size = hidden_size

    def build_chapter_head(self, output_size):
        # Cross Attention head
        self.fusion_head = ChapterHead(
            self.lang_embed_size, self.vision_embed_size, self.hidden_size, num_heads=8, output_size=output_size
        )

        # Window self attention head
        window_config = type('Config', (), {
            'hidden_size': self.hidden_size,
            'num_attention_heads': 8,
            'attention_probs_dropout_prob': 0.1
        })
        self.window_head = VideoChapterClassifier(window_config)


    def forward(self, img_clip, text_ids, attention_mask, return_emb=False):
        batch_size, num_clips, seq_length = text_ids.shape

        # Process language embeddings clip by clip
        clip_fusion_embs = []

        for i in range(num_clips):
            # Language processing
            clip_text_ids = text_ids[:, i, :].reshape(batch_size, -1)   # [batch_size, seq_length] 
            clip_attn_mask = attention_mask[:, i, :]
            lang_output = self.lang_model(input_ids=clip_text_ids, attention_mask=clip_attn_mask)
            lang_emb = lang_output["pooler_output"]  # [batch_size, hidden_size]
            # print(f'clip lang embedding: {lang_emb.shape}')

            # Vision processing
            with torch.cuda.amp.autocast():  # 메모리 효율을 위한 mixed precision
                clip_frames = img_clip[:, i]
                frames = rearrange(clip_frames, 'b nf c h w -> (b nf) c h w')
                frame_emb = self.vision_model(frames)
                frame_emb = frame_emb.view(batch_size, self.segment_size, -1).float()
                # print(f'clip vision embedding: {frame_emb.shape}')

            # Cross attention for current clip
            clip_fusion = self.fusion_head.process_clip(lang_emb, frame_emb) # [batch, hidden_size]
            # print(f'clip fusion embedding: {clip_fusion.shape}')
            # clip_fusion_embs.append(clip_fusion)
            clip_fusion_embs.append(clip_fusion.detach().cpu())

        # all_fusion_embs = torch.stack(clip_fusion_embs, dim=1)
        all_fusion_embs = torch.stack([emb.cuda() for emb in clip_fusion_embs], dim=1) # [batch, num_clips, hidden_size]
        # print(f'all_fusion_embs: {all_fusion_embs.shape}')
        
        binary_logits, binary_prob = self.window_head(all_fusion_embs)

        del lang_output, lang_emb, clip_frames, frames, frame_emb, clip_fusion, clip_fusion_embs
        gc.collect()
        torch.cuda.empty_cache()

        # binary_logits, binary_prob = classifier(fusion_emb)
        # binary_prob = F.softmax(binary_logits, dim=1)

        return binary_logits, binary_prob
        # return all_fusion_embs
    
    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

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