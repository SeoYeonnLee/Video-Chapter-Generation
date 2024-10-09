"""
pegasus text summarization model wrapped with vision embedding

"""

import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from common_utils.language_model_utils import top_k_logits


logger = logging.getLogger(__name__)


class VisualLangCrossAttention(nn.Module):
    """
    A vanilla multi-head attention layer with a projection at the end.
    """
    def __init__(self, n_embd, n_head, output_size, attn_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd

        # cross attention K,V,Q
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, output_size)
        
 
    def forward(self, query_states, key_value_states, kv_attention_mask=None):
        """
        query_states: language hidden states, shape: B, T1, C
        key_value_states: visual hidden states, shape: B, T2, C
        kv_attention_mask: mask some padding states in key_value_states, shape: B, T2
        """
        B, T1, C = query_states.size()
        B, T2, C = key_value_states.size()

        q = self.query(query_states).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(key_value_states).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(key_value_states).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # produces attention weight
        # (B, nh, T1, hs) x (B, nh, hs, T2) -> (B, nh, T1, T2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if kv_attention_mask is not None:
            att = kv_attention_mask.view(B, 1, 1, T2) * att
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v     # (B, nh, T1, T2) x (B, nh, T2, hs) -> (B, nh, T1, hs)
        y = y.transpose(1, 2).contiguous().view(B, T1, C)    # re-assemble all head outputs side by side

        # output projection
        out = self.proj(y)
        return out


class FusionHead(nn.Module):
    def __init__(self, lang_emb_size, vision_emb_size, hidden_size, fusion_type="mlp"):
        super(FusionHead, self).__init__()
        self.lang_emb_size = lang_emb_size
        self.vision_emb_size = vision_emb_size
        self.hidden_size = hidden_size
        self.fusion_type = fusion_type
        self.lang_proj_head = nn.Linear(lang_emb_size, hidden_size, bias=False)
        self.vision_proj_head = nn.Linear(vision_emb_size, hidden_size, bias=False)

        if self.fusion_type == "mlp":
            self.fusion_head = nn.Linear(2 * hidden_size, lang_emb_size, bias=False)
        else:
            self.fusion_head = VisualLangCrossAttention(hidden_size, 8, lang_emb_size, attn_pdrop=0)

    def forward(self, lang_emb, vision_emb, vision_attention_mask=None):
        """
        lang_emb: [batch, T1, lang_emb_size], e.g. 8, 512, 1024
        vision_emb: [batch, T2, vision_emb] e.g. 8, 10, 2048
        vision_attention_mask: [batch, T2]
        """

        lang_out = self.lang_proj_head(lang_emb)       # batch, T1, hidden_size
        lang_out = F.relu(lang_out)

        vision_out = self.vision_proj_head(vision_emb)  # batch, T2, hidden_size
        vision_out = F.relu(vision_out)

        if self.fusion_head == "mlp":
            vision_out = torch.sum(vision_out, dim=1)
            valid_num = torch.sum(vision_attention_mask, dim=1).view(-1, 1)
            vision_out = vision_out / valid_num

            vision_out = vision_out.unsqueeze(1)
            vision_out = vision_out.repeat((1, lang_out.size(1), 1))    # batch, T1, hidden_size
            fusion_emb = torch.cat([vision_out, lang_out], dim=2)       # batch, T1, 2 * hidden_size
            fusion_out = self.fusion_head(fusion_emb)                   # batch, T, hidden_size
        else:
            fusion_out = self.fusion_head(query_states=lang_out, key_value_states=vision_out, kv_attention_mask=vision_attention_mask)       

        return fusion_out


class PegasusVisionEmb(nn.Module):
    def __init__(self, reinit_head=True, fusion_type="mlp"):
        super().__init__()
        self.fusion_type = fusion_type
        # load pretrained base model
        self.base_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')

        self.encoder = self.base_model.get_encoder()
        self.decoder = self.base_model.get_decoder()
        self.lm_head = self.base_model.lm_head
        
        self.vocab_size = self.base_model.config.vocab_size
        self.embed_size = self.base_model.config.hidden_size
        if fusion_type == "mlp":
            self.fusion_head = FusionHead(self.embed_size, 2048, 128, fusion_type)
        else:
            self.fusion_head = FusionHead(self.embed_size, 2048, self.embed_size, fusion_type)

        # init head
        if reinit_head:
            self.base_model.lm_head = nn.Linear(self.embed_size, self.vocab_size, bias=False)
            self.base_model.lm_head.weight.data.normal_(mean=0.0, std=0.02)
            if self.base_model.lm_head.bias is not None:
                self.base_model.lm_head.bias.data.zero_()

        # number of parameters
        print("number of parameters: ", sum(p.numel() for p in self.parameters()))

    def fix_backbone(self):
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        for pn, p in self.named_parameters():
            if "pooler" in pn or "head" in pn:
                continue
            p.requires_grad = False

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

        # for mn, m in self.named_modules():
        for pn, p in self.named_parameters():
            # fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            fpn = pn
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif "layer_norm" in fpn:
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
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in list(decay)], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in list(no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, vision_emb, vision_attention_mask, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        # fusion vision emb and text emb here
        fusion_out = self.fusion_head(encoder_outputs[0], vision_emb, vision_attention_mask)
        fusion_out = fusion_out + encoder_outputs[0]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fusion_out,
            encoder_attention_mask=attention_mask,
            return_dict=True
        )

        logits = self.lm_head(decoder_outputs.last_hidden_state) + self.base_model.final_logits_bias

        return logits
    

    def generate(self, vision_emb, vision_attention_mask, text, tokenizer, device, max_text_len=512, max_len=30, temperature=1.0, sample=False, top_k=None):
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:max_text_len]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.from_numpy(np.array(ids)).long().to(device)
        input_ids = input_ids.unsqueeze(0)
        
        decoder_start_token_id = self.base_model.config.decoder_start_token_id
        # decoder_start_token = tokenizer.convert_ids_to_tokens([decoder_start_token_id])
        decoder_input_ids = torch.from_numpy(np.array([decoder_start_token_id])).long().to(device)
        decoder_input_ids = decoder_input_ids.unsqueeze(0)

        i = 0
        sentence_ids = []
        sentence_logits = []
        while i < max_len:
            with torch.no_grad():
                logits = self.forward(vision_emb, vision_attention_mask, input_ids, decoder_input_ids=decoder_input_ids)
            sentence_logits.append(logits)

            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            decoder_input_ids = torch.cat((decoder_input_ids, ix), dim=1)
            sentence_ids.append(ix.squeeze(0).item())

            if sentence_ids[-1] == self.base_model.config.eos_token_id:
                break
            i += 1
        
        sentence_logits = torch.cat(sentence_logits, dim=1)
        gen_text = tokenizer.decode(sentence_ids)
        return gen_text, sentence_logits


if __name__ == "__main__":
    model = PegasusHugface()
    print(model)

    class TrainerConfig:
        # optimization parameters
        max_epochs = 10
        block_size = 512
        batch_size = 64
        learning_rate = 3e-4
        betas = (0.9, 0.95)
        grad_norm_clip = 1.0
        weight_decay = 0.01  # only applied on matmul weights
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        lr_decay = False
        lr_decay_type = "cosine"
        warmup_epochs = 30 
        final_epoch = 2700 
        # checkpoint settings
        ckpt_path = None
        num_workers = 0    # for DataLoader
        # tensorboard writer
        tensorboard_writer = None

        def __init__(self, **kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)
        
    config = TrainerConfig()
    model.configure_optimizers(config)

    src_text = ["PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."]

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large').to(device)

    print(model)

    tokens = tokenizer.tokenize(src_text[0])
    ids = tokenizer.convert_tokens_to_ids(tokens)

    batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=False)

    print(src_text[0])
    print(tgt_text)


