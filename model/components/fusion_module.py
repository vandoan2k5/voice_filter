import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class FusionModule(nn.Module):
    def __init__(self,
                 emb_dim,
                 nhead=4,
                 dropout=0.1):
        super(FusionModule, self).__init__()
        self.nhead = nhead
        self.dropout = dropout
        param_size = [1, 1, emb_dim]

        self.attn = nn.MultiheadAttention(emb_dim,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          batch_first=True)
        self.fusion = nn.Conv2d(emb_dim * 2, emb_dim, kernel_size=1)
        self.alpha = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.zeros_(self.alpha)

    def forward(self,
                aux: torch.Tensor,
                esti: torch.Tensor) -> torch.Tensor:
        aux = aux.unsqueeze(1)  # [B, 1, C]
        flatten_esti = esti.flatten(start_dim=2).transpose(1, 2)  # [B, T*F, C]
        aux_adapt = self.attn(aux, flatten_esti, flatten_esti, need_weights=False)[0]
        aux = aux + self.alpha * aux_adapt  # [B, 1, C]

        aux = aux.unsqueeze(-1).transpose(1, 2).expand_as(esti)
        esti = self.fusion(torch.cat((esti, aux), dim=1))  # [B, C, T, F]

        return esti