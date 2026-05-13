from typing import Tuple
import torch
import torch.nn as nn
from .components import AuxEncoder, FusionModule, GridNetBlock, EnUnetModule, LayerNormalization4D, LayerNormalization4DCF, GateConv2d, Conv2dUnit, Deconv2dUnit

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
                 num_spks=101,
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

        # Speaker Encoder
        self.aux_encoder = AuxEncoder(emb_dim, num_spks)

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
                aux: torch.Tensor,
                aux_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.
            Args:
                mix (torch.Tensor): batched audio tensor with N samples [B, 2, T, F]
                aux (torch.Tensor): batched audio tensor with N samples [B, T, F]
                aux_lengths (torch.Tensor): aux input lengths [B]
            Returns:
                enhanced (torch.Tensor): [B, 2, T, F] audio tensors with N samples.
        """
        # Speech Encoder
        esti = self.conv(mix)  # [B, -1, T, F]
        aux = self.conv(aux)  # [B, -1, T, F]

        # Speaker Encoder
        aux, speak_pred = self.aux_encoder(aux, aux_lengths)

        # Speaker Extractor
        for i in range(self.n_layers):
            esti = self.fusion_blocks[i](aux, esti)
            esti = self.separate_blocks[i](esti)  # [B, -1, T, F]

        # Speech Decoder
        esti = self.deconv(esti)  # [B, 2, T, F]

        return esti, speak_pred


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
                     num_spks=configs['path']['num_spks'],
                     activation=configs['net']['activation'],
                     eps=configs['net']['eps']).to(device)

    # num_params = sum([param.nelement() for param in net.parameters()]) / 10.0 ** 6
    # print(num_params)

    # aux = torch.ones((2, 2, 501, 129), device=device)
    # inpt = torch.ones((2, 2, 501, 129), device=device)
    # aux_len = torch.tensor([501, 401], dtype=torch.int, device=device)
    # output, spk_pred = net(inpt, aux, aux_len)
    # print(output.shape)

    from ptflops import get_model_complexity_info
    
    
    def input_constructor(input_shape):
        inputs = {'mix': torch.ones((1, *input_shape), device=device),
                  'aux': torch.ones((1, *input_shape), device=device),
                  'aux_lengths': torch.tensor([input_shape[1]], dtype=torch.int, device=device)}

        return inputs
    
    macs, params = get_model_complexity_info(net, (2, 501, 129),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             input_constructor=input_constructor,
                                             verbose=True,
                                             output_precision=4)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
