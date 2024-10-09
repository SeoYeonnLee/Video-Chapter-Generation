import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig

logger = logging.getLogger(__name__)


class BertHugfaceConstrast(nn.Module):
    def __init__(self, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = BertModel.from_pretrained('bert-base-uncased')
        self.encoder_k = BertModel.from_pretrained('bert-base-uncased')

        self.vocab_size = self.encoder_q.config.vocab_size
        self.embed_size = self.encoder_q.config.hidden_size

        # create the queue
        self.register_buffer("queue", torch.randn(self.embed_size, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # number of parameters: 109482240
        print("number of parameters: ", sum(p.numel() for p in self.encoder_q.parameters()))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    

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

    def forward(self, query_clip_text_id, query_att_mask, pos_candidates_text_id, pos_candidates_att_mask, device):
        batch_size, cand_size, text_size = pos_candidates_text_id.size()

        # encode querys
        inputs_q = {
            "input_ids": query_clip_text_id,
            "attention_mask": query_att_mask
        }
        output_q = self.encoder_q(**inputs_q)
        clip_emb_q = output_q.pooler_output
        clip_emb_q = F.normalize(clip_emb_q, dim=1)     # normalize 
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

            # encode candidates for selecting positive example
            pos_candidates_text_id = pos_candidates_text_id.view(-1, text_size)  
            pos_candidates_att_mask = pos_candidates_att_mask.view(-1, text_size)
            inputs_cand = {
                "input_ids": pos_candidates_text_id,
                "attention_mask": pos_candidates_att_mask
            }
            output_cand = self.encoder_q(**inputs_cand)
            clip_emb_cand = output_cand.pooler_output
            clip_emb_cand = F.normalize(clip_emb_cand, dim=1)     # normalize 

            # restore size
            pos_candidates_text_id = pos_candidates_text_id.view(batch_size, cand_size, text_size)
            pos_candidates_att_mask = pos_candidates_att_mask.view(batch_size, cand_size, text_size)

            # find the max logits as positive example
            clip_emb_cand = clip_emb_cand.view(batch_size, cand_size, self.embed_size)
            clip_emb_q_dummy = clip_emb_q.unsqueeze(2)
            k = torch.bmm(clip_emb_cand, clip_emb_q_dummy).squeeze(2)
            k0 = torch.argmax(k, dim=1)
            
            gather_idx = k0.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, text_size)
            select_pos_candidates_text_id = torch.gather(pos_candidates_text_id, dim=1, index=gather_idx).squeeze(1)
            select_pos_candidates_att_mask = torch.gather(pos_candidates_att_mask, dim=1, index=gather_idx).squeeze(1)

            # encode keys
            inputs_k = {
                "input_ids": select_pos_candidates_text_id,
                "attention_mask": select_pos_candidates_att_mask
            }
            output_k = self.encoder_k(**inputs_k)  
            clip_emb_k = output_k.pooler_output
            clip_emb_k = F.normalize(clip_emb_k, dim=1)     # normalize 

        # compute logits by using einstein sum
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [clip_emb_q, clip_emb_k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [clip_emb_q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(clip_emb_k)

        return logits, labels



if __name__ == "__main__":
    model = BertHugfaceConstrast()
    print(model)

