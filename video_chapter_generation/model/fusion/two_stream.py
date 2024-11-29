import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math


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
    def __init__(self, lang_emb_size, vision_emb_size, segment_size, hidden_size, output_size, head_type="mlp"):
        super(ChapterHead, self).__init__()
        self.lang_emb_size = lang_emb_size
        self.vision_emb_size = vision_emb_size
        self.segment_size = segment_size
        self.hidden_size = hidden_size
        self.head_type = head_type

        self.lang_proj_head = nn.Linear(lang_emb_size, hidden_size, bias=False)
        self.vision_proj_head = nn.Linear(vision_emb_size, hidden_size, bias=False)

        if head_type == "mlp":
            self.head = nn.Linear((segment_size + 1) * hidden_size, output_size, bias=True)
        elif head_type == "attn":
            self.head = SelfAttention(hidden_size, 4, output_size)
        else:
            raise RuntimeError(f"Unknown head_type {head_type}")


    def forward(self, lang_emb, vision_emb):
        """
        lang_emb: [batch, lang_emb_size]
        vision_emb: [batch, segment_size, vision_emb]
        """

        batch_size = lang_emb.shape[0]

        lang_out = self.lang_proj_head(lang_emb).unsqueeze(1)        # batch, 1, hidden_size
        lang_out = F.relu(lang_out)
        # print(f'lang_out shape: {lang_out.shape}')

        vision_emb = vision_emb.view(-1, self.vision_emb_size)
        vision_out = self.vision_proj_head(vision_emb).view(batch_size, self.segment_size, self.hidden_size)  # batch, segment, hidden_size
        vision_out = F.relu(vision_out)
        # print(f'vision_out shape: {vision_out.shape}')

        fusion_emb = torch.cat([vision_out, lang_out], dim=1)
        # print(f'fusion_emb shape: {fusion_emb.shape}') 
        if self.head_type == "mlp":
            fusion_emb = fusion_emb.view(batch_size, -1)
        out = self.head(fusion_emb)
        # print(f'out shape: {out.shape}')

        return out



class TwoStream(nn.Module):
    def __init__(self, lang_model, vision_model, lang_embed_size, vision_embed_size, segment_size, hidden_size):
        super(TwoStream, self).__init__()
        self.lang_model = lang_model
        self.vision_model = vision_model
        self.segment_size = segment_size
        self.lang_embed_size = lang_embed_size
        self.vision_embed_size = vision_embed_size
        self.hidden_size = hidden_size

    # def build_chapter_head(self, input_size, output_size):
    #     """
    #     build a new head for video chapter prediction
    #     """
    #     self.head = nn.Linear(input_size, output_size, bias=True)
    #     self.head.weight.data.normal_(mean=0.0, std=0.02)
    #     if self.head.bias is not None:
    #         self.head.bias.data.zero_()

    def build_chapter_head(self, output_size, head_type="mlp"):
        """
        build a new head for video chapter prediction
        head_type: mlp or attn
        """

        self.fusion_head = ChapterHead(self.lang_embed_size, self.vision_embed_size, self.segment_size, self.hidden_size, output_size, head_type)


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

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
        

    def forward(self, img_clip, text_ids, attention_mask, return_emb=False):
        # language
        lang_model_inputs = {
            "input_ids": text_ids,
            "attention_mask": attention_mask
        }
        lang_model_output = self.lang_model(**lang_model_inputs)
        lang_emb = lang_model_output.pooler_output

        # vision
        batch_size = img_clip.shape[0]
        img_clip = rearrange(img_clip, 'b t c h w -> (b t) c h w').contiguous()
        vision_emb = self.vision_model(img_clip)
        vision_emb = vision_emb.view(batch_size, self.segment_size, -1)

        # fusion
        binary_logits = self.fusion_head(lang_emb, vision_emb)
        binary_prob = F.softmax(binary_logits, dim=1)

        if return_emb:
            return binary_logits, binary_prob, vision_emb, lang_emb
        
        return binary_logits, binary_prob
