"""
apply listnet loss for video segment point prediction
"""


import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig


logger = logging.getLogger(__name__)


class BertHugface(nn.Module):
    def __init__(self, pretrain_stage=True):
        super().__init__()
        self.pretrain_stage = pretrain_stage
        # load pretrained base model
        if self.pretrain_stage:
            self.base_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        else:
            # config = BertConfig(hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5, output_attentions=True)
            # self.base_model = BertModel.from_pretrained('bert-base-uncased', config=config)
            self.base_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

        self.vocab_size = self.base_model.config.vocab_size
        self.embed_size = self.base_model.config.hidden_size

        # init head
        self.head = nn.Linear(self.embed_size, self.vocab_size, bias=False)
        self.head.weight.data.normal_(mean=0.0, std=0.02)
        if self.head.bias is not None:
            self.head.bias.data.zero_()

        # number of parameters: 109482240
        print("number of parameters: ", sum(p.numel() for p in self.base_model.parameters()))

    def build_chapter_head(self):
        """
        build a new head for video chapter prediction
        """
        self.surrogate_head = nn.Linear(self.embed_size, 1, bias=True)
        self.surrogate_head.weight.data.normal_(mean=0.0, std=0.02)
        if self.surrogate_head.bias is not None:
            self.surrogate_head.bias.data.zero_()
        
        self.head = nn.Linear(self.embed_size, 2, bias=True)
        self.head.weight.data.normal_(mean=0.0, std=0.02)
        if self.head.bias is not None:
            self.head.bias.data.zero_()

    
    def load_constrast_checkpoint(self, checkpoint_path):
        from collections import OrderedDict
        constrast_state_dict = torch.load(checkpoint_path)
        encoder_state = OrderedDict()
        for k, v in constrast_state_dict.items():
            if "encoder_q" in k:
                key = k[k.index(".") + 1:]
                encoder_state[key] = v
        self.base_model.load_state_dict(encoder_state)

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
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif "ln" in fpn:
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

    def train_forward(self, x, attention_mask, targets, balance_selected_indices, binary_cls_label, get_attention=False):
        """
        x: [batch_size, slate_length, text_length]
        attention_mask: [batch_size, slate_length, text_length]
        targets: [batch_size, slate_length]
        """
        batch_size, slate_length, text_length = x.size()
        x = x.view(-1, text_length)
        attention_mask = attention_mask.view(-1, text_length)
        inputs = {
            "input_ids": x,
            "attention_mask": attention_mask
        }
        output = dict()

        base_output = self.base_model(**inputs)
        
        # when pretrain, it runs MLM self-supervised task
        if self.pretrain_stage:
            last_hidden_state = base_output.last_hidden_state
            logits = self.head(last_hidden_state)

            if targets is not None:
                # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                mask = torch.nonzero(targets != -1)
                valid_logits = logits[mask[:, 0], mask[:, 1], :]
                valid_targets = targets[mask[:, 0], mask[:, 1]]
                loss = F.cross_entropy(valid_logits.view(-1, valid_logits.size(-1)), valid_targets.view(-1))
                output["logits"] = logits
                output["loss"] = loss

        else:
            pool_output = base_output.pooler_output                         # [batch * slate_length, hidden_size]

            # surrogate_logits = self.surrogate_head(pool_output)             # [batch * slate_length, 1]
            # surrogate_logits = surrogate_logits.view(batch_size, slate_length)
            # prob = F.softmax(surrogate_logits, dim=1)
            # preds_smax = prob + 1e-10
            # preds_log = torch.log(preds_smax)
            # surrogate_loss = torch.mean(-torch.sum(targets * preds_log, dim=1))

            all_hidden_embedding = pool_output.view(batch_size, slate_length, self.embed_size)      # [batch, slate_length, hidden_size]
            pos_emb = all_hidden_embedding[:, 0, :].unsqueeze(1)
            contrast_emb = all_hidden_embedding[:, 1:, :]
            contrast_emb = torch.transpose(contrast_emb, 1, 2)
            surrogate_logits = torch.bmm(pos_emb, contrast_emb)     # [batch, 1, slate_length-1]
            surrogate_logits = surrogate_logits.squeeze(1)
            prob = F.softmax(surrogate_logits, dim=1)
            preds_smax = prob + 1e-10
            preds_log = torch.log(preds_smax)
            targets = targets[:, 1:]
            surrogate_loss = torch.mean(-torch.sum(targets * preds_log, dim=1))

            binary_out = pool_output[balance_selected_indices]
            binary_logits = self.head(binary_out)
            binary_prob = F.softmax(binary_logits, dim=1)
            binary_loss = F.cross_entropy(binary_logits, binary_cls_label)

            output["binary_logits"] = binary_logits
            output["binary_prob"] = binary_prob
            output["loss"] = surrogate_loss + binary_loss

        if get_attention:
            output["bert_attention"] = base_output.attentions
            
        return output

    def test_forward(self, x, attention_mask, targets):
        inputs = {
            "input_ids": x,
            "attention_mask": attention_mask
        }
        output = dict()

        base_output = self.base_model(**inputs)
        pool_output = base_output.pooler_output
        binary_logits = self.head(pool_output)
        binary_prob = F.softmax(binary_logits, dim=1)
        binary_loss = F.cross_entropy(binary_logits, targets)

        output["binary_logits"] = binary_logits
        output["binary_prob"] = binary_prob
        output["loss"] = binary_loss
        return output


if __name__ == "__main__":
    model = BertHugface(pretrain_stage=False)
    model.fix_backbone()
    print(model)

