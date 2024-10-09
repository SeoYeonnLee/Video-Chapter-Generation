import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from ops.temporal_shift import make_temporal_shift
from ops.basic_ops import Identity
from einops import rearrange


class Resnet50TSM(torch.nn.Module):
    def __init__(self, segments_size=8, shift_div=8, pretrain_stage=True):
        super(Resnet50TSM, self).__init__()
        self.pretrain_stage = pretrain_stage
        
        self.base_model = torchvision.models.resnet50(pretrained=True)
        make_temporal_shift(self.base_model, n_segment=segments_size, n_div=shift_div)
        self.segments_size = segments_size
        self.feature_dim = self.base_model.fc.in_features
        self.base_model.fc = Identity()  # discard last fc layer
        self.head = None

    def build_chapter_head(self):
        self.head = nn.Linear(self.segments_size * self.feature_dim, 2)


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
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, 'b t c h w -> (b t) c h w').contiguous()
        out = self.base_model(x)
        out = out.view(batch_size, self.segments_size, -1)
        out = out.view(batch_size, -1)      # concatenate all vision embedding along with segment dim
        logits = self.head(out)
        prob = F.softmax(logits, dim=1)
        
        return logits, prob


if __name__ == "__main__":
    model = Resnet50TSM()
    print(model)
