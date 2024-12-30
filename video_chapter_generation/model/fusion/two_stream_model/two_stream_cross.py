import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math

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
        lang_out: [batch, num_clip, 1, hidden_size]
        vision_out: [batch, num_clip, num_frame, hidden_size]
        """
        batch_size, num_clip, _, hidden_size = lang_out.shape
        _, _, num_frame, _ = vision_out.shape

        # Project inputs
        query = self.query_proj(lang_out).view(batch_size * num_clip, 1, self.num_heads, self.head_dim)
        key = self.key_proj(vision_out).view(batch_size * num_clip, num_frame, self.num_heads, self.head_dim)
        value = self.value_proj(vision_out).view(batch_size * num_clip, num_frame, self.num_heads, self.head_dim)

        # Transpose for multi-head attention
        query = query.permute(0, 2, 1, 3)  # [batch * num_clip, num_heads, 1, head_dim]
        key = key.permute(0, 2, 3, 1)      # [batch * num_clip, num_heads, head_dim, num_frame]
        value = value.permute(0, 2, 1, 3)  # [batch * num_clip, num_heads, num_frame, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(query, key) / math.sqrt(self.head_dim)  # [batch * num_clip, num_heads, 1, num_frame]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch * num_clip, num_heads, 1, num_frame]
        attn_probs = self.dropout(attn_probs)

        # Apply attention to Value
        attn_output = torch.matmul(attn_probs, value)  # [batch * num_clip, num_heads, 1, head_dim]

        # Reshape and project back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch * num_clip, 1, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, num_clip, hidden_size)  # [batch, num_clip, hidden_size]
        fusion_emb = self.out_proj(attn_output)  # [batch, num_clip, hidden_size]

        return fusion_emb

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
    def __init__(self, lang_emb_size, vision_emb_size, hidden_size, num_heads, output_size):
        super().__init__()
        self.lang_proj_head = nn.Linear(lang_emb_size, hidden_size)
        self.vision_proj_head = nn.Linear(vision_emb_size, hidden_size)
        self.cross_attention = CrossAttention(hidden_size, num_heads=8)
        self.final_mlp = nn.Linear(hidden_size, output_size)

    def forward(self, lang_out, vision_out):
        """
        lang_out: [batch, num_clip, lang_emb_size]
        vision_out: [batch, num_clip, num_frame, vision_emb_size]
        """
        batch_size, num_clip, num_frame, _ = vision_out.shape

        # Project `vision_out` to hidden size
        vision_out_flat = vision_out.view(-1, vision_out.size(-1))  # Flatten for projection
        vision_out_proj = self.vision_proj_head(vision_out_flat)  # [batch*num_clip*num_frame, hidden_size]
        vision_out_proj = vision_out_proj.view(batch_size, num_clip, num_frame, -1)  # Reshape
        vision_out_proj = F.relu(vision_out_proj)  # Apply activation

        print(f"Projected vision_out shape: {vision_out_proj.shape}")  # [batch, num_clip, num_frame, hidden_size]

        # Project `lang_out` to hidden size
        lang_out_proj = self.lang_proj_head(lang_out)  # [batch, num_clip, hidden_size]
        lang_out_proj = F.relu(lang_out_proj).unsqueeze(2)  # Add frame dimension: [batch, num_clip, 1, hidden_size]

        print(f"Projected lang_out shape: {lang_out_proj.shape}")  # [batch, num_clip, 1, hidden_size]

        # Apply cross-attention
        fusion_emb = self.cross_attention(lang_out_proj, vision_out_proj)  # [batch, num_clip, hidden_size]
        print(f"Fusion embedding shape: {fusion_emb.shape}")

        # Pooling
        pooled_emb = fusion_emb.mean(dim=1)  # Average across clips: [batch, hidden_size]
        print(f"Pooled embedding shape: {pooled_emb.shape}")

        # Final classification
        output = self.final_mlp(pooled_emb)  # [batch, output_size]
        print(f"Final output shape: {output.shape}")

        return output

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
        self.fusion_head = ChapterHead(
            self.lang_embed_size, self.vision_embed_size, self.hidden_size, num_heads=8, output_size=output_size
        )

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

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

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                           % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, img_clip, text_ids, attention_mask, return_emb=False):
        batch_size, num_clips, seq_length = text_ids.shape

        # Process language embeddings
        text_ids = text_ids.view(batch_size * num_clips, seq_length)
        attention_mask = attention_mask.view(batch_size * num_clips, seq_length)
        lang_model_output = self.lang_model(input_ids=text_ids, attention_mask=attention_mask)
        lang_emb = lang_model_output["pooler_output"]  # [batch_size * num_clips, hidden_size]
        lang_emb = lang_emb.view(batch_size, num_clips, -1)  # [batch_size, num_clips, hidden_size]

        # Process vision embeddings
        img_clip = rearrange(img_clip, 'b nc nf c h w -> (b nc nf) c h w')
        vision_emb = self.vision_model(img_clip)
        vision_emb = vision_emb.view(batch_size, num_clips, self.segment_size, -1)

        # Fusion and classification
        binary_logits = self.fusion_head(lang_emb, vision_emb)
        binary_prob = F.softmax(binary_logits, dim=1)

        if return_emb:
            return binary_logits, binary_prob, vision_emb, lang_emb

        return binary_logits, binary_prob

'''class TwoStream(nn.Module):
    def __init__(self, lang_model, vision_model, lang_embed_size, vision_embed_size, segment_size, hidden_size):
        super(TwoStream, self).__init__()
        self.lang_model = lang_model
        self.vision_model = vision_model
        self.segment_size = segment_size
        self.lang_embed_size = lang_embed_size
        self.vision_embed_size = vision_embed_size
        self.hidden_size = hidden_size

    def build_chapter_head(self, output_size):
        self.fusion_head = ChapterHead(
            self.lang_embed_size, self.vision_embed_size, self.hidden_size, num_heads=8, output_size=output_size
        )

    def forward(self, img_clip, text_ids, attention_mask, return_emb=False):
        batch_size, num_clips, seq_length = text_ids.shape

        # Process language embeddings
        text_ids = text_ids.view(batch_size * num_clips, seq_length)
        attention_mask = attention_mask.view(batch_size * num_clips, seq_length)
        lang_model_output = self.lang_model(input_ids=text_ids, attention_mask=attention_mask)
        lang_emb = lang_model_output["pooler_output"]  # [batch_size * num_clips, hidden_size]
        lang_emb = lang_emb.view(batch_size, num_clips, -1)  # [batch_size, num_clips, hidden_size]

        # Process vision embeddings
        img_clip = rearrange(img_clip, 'b nc nf c h w -> (b nc nf) c h w')
        vision_emb = self.vision_model(img_clip)
        vision_emb = vision_emb.view(batch_size, num_clips, self.segment_size, -1)

        # Fusion and classification
        binary_logits = self.fusion_head(lang_emb, vision_emb)
        binary_prob = F.softmax(binary_logits, dim=1)

        if return_emb:
            return binary_logits, binary_prob, vision_emb, lang_emb

        return binary_logits, binary_prob
'''

if __name__ == "__main__":
    from transformers import BertModel, BertConfig
    from torchvision.models import resnet50
    import torch

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_clips = 4  # Number of clips per batch
    num_frames = 8  # Number of frames per clip
    img_height, img_width = 224, 224
    seq_length = 100  # Token sequence length
    hidden_size = 128  # Hidden size for fusion
    lang_embed_size = 768  # BERT's hidden size
    vision_embed_size = 2048  # ResNet50's output size
    output_size = 2  # Binary classification

    # Dummy language model (BERT)
    class DummyLangModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = BertModel(BertConfig(hidden_size=lang_embed_size, num_hidden_layers=2))
            self.pooler = nn.Linear(lang_embed_size, lang_embed_size)

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooler_output = self.pooler(outputs.last_hidden_state[:, 0])
            return {"pooler_output": pooler_output}

    # Dummy vision model (ResNet50)
    class DummyVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = resnet50()
            self.model.fc = nn.Identity()  # Remove the final classification layer

        def forward(self, x):
            return self.model(x)

    # Initialize models
    lang_model = DummyLangModel().to(device)
    vision_model = DummyVisionModel().to(device)

    # Initialize TwoStream model
    two_stream_model = TwoStream(
        lang_model=lang_model,
        vision_model=vision_model,
        lang_embed_size=lang_embed_size,
        vision_embed_size=vision_embed_size,
        segment_size=num_frames,
        hidden_size=hidden_size,
    )
    two_stream_model.build_chapter_head(output_size=output_size)
    two_stream_model = two_stream_model.to(device)

    # Generate dummy data
    img_clip = torch.randn(batch_size, num_clips, num_frames, 3, img_height, img_width).to(device)  # Image clips
    text_ids = torch.randint(0, 30522, (batch_size, num_clips, seq_length)).to(device)  # Random token IDs
    attention_mask = torch.ones_like(text_ids).to(device)  # Attention mask

    # Forward pass
    with torch.no_grad():
        binary_logits, binary_prob = two_stream_model(img_clip, text_ids, attention_mask)

        print("\n=== Debugging Outputs ===")
        print(f"Binary logits: {binary_logits}")
        print(f"Binary probabilities: {binary_prob}")
