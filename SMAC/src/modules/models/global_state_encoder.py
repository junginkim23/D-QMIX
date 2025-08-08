import torch
import torch.nn as nn
import torch.nn.functional as F


class Global_State_Encoder(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 args):
        super(Global_State_Encoder, self).__init__()

        self.args = args
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(in_features, out_features, bias=True))

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

