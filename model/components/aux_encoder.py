from typing import Tuple
import torch
import torch.nn as nn

from .enunet_module import EnUnetModule


class AuxEncoder(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_spks):
        super(AuxEncoder, self).__init__()
        k1, k2 = (1, 3), (1, 3)
        self.d_feat = emb_dim

        self.aux_enc = nn.ModuleList([EnUnetModule(emb_dim, emb_dim, (1, 5), k2, scale=4),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=3),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=2),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=1)])
        self.out_conv = nn.Linear(emb_dim, emb_dim)
        self.speaker = nn.Linear(emb_dim, num_spks)

    def forward(self,
                auxs: torch.Tensor,
                aux_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aux_lengths = (((aux_lengths // 3) // 3) // 3) // 3

        for i in range(len(self.aux_enc)):
            auxs = self.aux_enc[i](auxs)  # [B, C, T, F]

        auxs = torch.stack([torch.mean(
            aux[:, :aux_length, :], dim=(1, 2)) for aux, aux_length in zip(auxs, aux_lengths)], dim=0)  # [B, C]
        auxs = self.out_conv(auxs)

        return auxs, self.speaker(auxs)