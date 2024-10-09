import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import OpenAIGPTModel

logger = logging.getLogger(__name__)


class GPTHugface(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self):
        super().__init__()
        # load pretrained base model
        self.base_model = OpenAIGPTModel.from_pretrained('openai-gpt')
        self.vocab_size = self.base_model.tokens_embed.num_embeddings
        self.embed_size = self.base_model.tokens_embed.embedding_dim

        # init head
        self.head = nn.Linear(self.embed_size, self.vocab_size, bias=False)
        self.head.weight.data.normal_(mean=0.0, std=0.02)
        if self.head.bias is not None:
            self.head.bias.data.zero_()

        # number of parameters: 147621888
        print("number of parameters: ", sum(p.numel() for p in self.base_model.parameters()))

    def build_chapter_head(self, output_size):
        """
        build a new head for video chapter prediction
        """
        self.head = nn.Linear(self.embed_size, output_size, bias=True)
        self.head.weight.data.normal_(mean=0.0, std=0.02)
        if self.head.bias is not None:
            self.head.bias.data.zero_()

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

    def forward(self, x, attention_mask=None, targets=None):
        inputs = {
            "input_ids": x,
            "attention_mask": attention_mask,
        }
        output = self.base_model(**inputs)
        last_hidden_state = output.last_hidden_state
        logits = self.head(last_hidden_state)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            mask = torch.nonzero(targets != -1)
            valid_logits = logits[mask[:, 0], mask[:, 1], :]
            valid_targets = targets[mask[:, 0], mask[:, 1]]
            loss = F.cross_entropy(valid_logits.view(-1, valid_logits.size(-1)), valid_targets.view(-1))
            
        return logits, loss



if __name__ == "__main__":
    model = GPTHugface()
    print(model)