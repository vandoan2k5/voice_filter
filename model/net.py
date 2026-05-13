from typing import Tuple
import torch
import torch.nn as nn
from .components import FusionModule, GridNetBlock


class pTFGridNet(nn.Module):
    def __init__(self,
                 n_fft=256,
                 n_layers=6,
                 lstm_hidden_units=256,
                 attn_n_head=4,
                 attn_approx_qk_dim=516,
                 emb_dim=32,
                 emb_ks=8,
                 emb_hs=1,
                 spk_emb_dim=192,
                 activation="prelu",
                 eps=1.0e-5):
        super().__init__()
        self.n_layers = n_layers
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

        # Speech Encoder
        t_ksize, f_ksize = 3, 3
        ks, padding = (t_ksize, f_ksize), (t_ksize // 2, f_ksize // 2)
        self.conv = nn.Sequential(nn.Conv2d(2, emb_dim, ks, padding=padding),
                                  nn.GroupNorm(1, emb_dim, eps=eps))

        # Speaker Embedding Projection
        self.spk_proj = nn.Sequential(nn.Linear(spk_emb_dim, emb_dim),
                                       nn.PReLU(emb_dim),
                                       nn.Linear(emb_dim, emb_dim))

        # Speaker Extractor
        self.fusion_blocks = nn.ModuleList([FusionModule(emb_dim) for _ in range(n_layers)])
        self.separate_blocks = nn.ModuleList([GridNetBlock(emb_dim,
                                                            emb_ks,
                                                            emb_hs,
                                                            n_freqs,
                                                            lstm_hidden_units,
                                                            n_head=attn_n_head,
                                                            approx_qk_dim=attn_approx_qk_dim,
                                                            activation=activation,
                                                            eps=eps) for _ in range(n_layers)])

        # Speech Decoder
        self.deconv = nn.ConvTranspose2d(emb_dim, 2, ks, padding=padding)

    def forward(self,
                mix: torch.Tensor,
                spk_emb: torch.Tensor) -> torch.Tensor:
        esti = self.conv(mix)

        spk_emb = self.spk_proj(spk_emb)

        for i in range(self.n_layers):
            esti = self.fusion_blocks[i](spk_emb, esti)
            esti = self.separate_blocks[i](esti)

        esti = self.deconv(esti)

        return esti


if __name__ == "__main__":
    import toml

    configs = toml.load('configs/train_config.toml')
    gpuids = tuple(configs['gpu']['gpu_ids'])
    device = torch.device("cuda:{}".format(gpuids[0]))

    net = pTFGridNet(n_fft=configs['signal']['fft_num'],
                     n_layers=configs['net']['n_layers'],
                     lstm_hidden_units=configs['net']['lstm_hidden_units'],
                     attn_n_head=configs['net']['attn_n_head'],
                     attn_approx_qk_dim=configs['net']['attn_approx_qk_dim'],
                     emb_dim=configs['net']['emb_dim'],
                     emb_ks=configs['net']['emb_ks'],
                     emb_hs=configs['net']['emb_hs'],
                     activation=configs['net']['activation'],
                     eps=configs['net']['eps']).to(device)

    from ptflops import get_model_complexity_info

    def input_constructor(input_shape):
        inputs = {'mix': torch.ones((1, *input_shape), device=device),
                  'spk_emb': torch.ones((1, 192), device=device)}
        return inputs

    macs, params = get_model_complexity_info(net, (2, 501, 129),
                                              as_strings=True,
                                              print_per_layer_stat=True,
                                              input_constructor=input_constructor,
                                              verbose=True,
                                              output_precision=4)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
