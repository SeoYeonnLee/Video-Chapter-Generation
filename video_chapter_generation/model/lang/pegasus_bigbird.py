"""
pegasus bigbird text summarization model

"""

import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PegasusTokenizer, BigBirdPegasusForConditionalGeneration, BigBirdPegasusConfig
from common_utils.language_model_utils import top_k_logits


logger = logging.getLogger(__name__)


class PegasusBigBirdHugface(nn.Module):
    def __init__(self, reinit_head=False):
        super().__init__()
        # load pretrained base model
        self.base_model = BigBirdPegasusForConditionalGeneration.from_pretrained('google/bigbird-pegasus-large-arxiv', output_attentions=True)
        self.base_model.get_encoder().set_attention_type("original_full")

        self.vocab_size = self.base_model.config.vocab_size
        self.embed_size = self.base_model.config.hidden_size

        # init head
        if reinit_head:
            self.base_model.lm_head = nn.Linear(self.embed_size, self.vocab_size, bias=False)
            self.base_model.lm_head.weight.data.normal_(mean=0.0, std=0.02)
            if self.base_model.lm_head.bias is not None:
                self.base_model.lm_head.bias.data.zero_()

        # number of parameters: 109482240
        print("number of parameters: ", sum(p.numel() for p in self.base_model.parameters()))

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

    def forward(self, x, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None):
        """
        We only finetune the pretrained model on abstractive summarization generation, without MLM task
        """
        inputs = {
            "input_ids": x,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask
        }
        base_output = self.base_model(**inputs)
        logits = base_output.logits

        return logits
    

    def generate(self, text, tokenizer, device, max_text_len=512, max_len=30, temperature=1.0, sample=False, top_k=None):
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
                logits = self.forward(input_ids, decoder_input_ids=decoder_input_ids)
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
    model = PegasusBigBirdHugface()
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

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = PegasusTokenizer.from_pretrained('google/bigbird-pegasus-large-arxiv')
    model = BigBirdPegasusForConditionalGeneration.from_pretrained('google/bigbird-pegasus-large-arxiv').to(device)

    print(model)

    tokens = tokenizer.tokenize(src_text[0])
    ids = tokenizer.convert_tokens_to_ids(tokens)

    batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=False)

    print(src_text[0])
    print(tgt_text)


    # tokenizer = PegasusTokenizer.from_pretrained('google/bigbird-pegasus-large-arxiv')
    # model = BigBirdPegasusForConditionalGeneration.from_pretrained('google/bigbird-pegasus-large-arxiv')

    # ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    # inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=4096, return_tensors='pt', truncation=True)

    # summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
    # print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])


