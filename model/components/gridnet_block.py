import math
import torch
import torch.nn as nn
import difflib

from .layernorm import LayerNormalization4D, LayerNormalization4DCF


def get_layer(l_name,
              library=torch.nn):
    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError("Layer with name {} not found in {}.\n Closest matches: {}".format(
            l_name, str(library), close_matches))
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError("Multiple matchs for layer with name {} not found in {}.\n All matches: {}".format(
            l_name, str(library), close_matches))
    else:
        layer_handler = getattr(library, match[0])

        return layer_handler

class GridNetBlock(nn.Module):
    def __init__(self,
                 emb_dim,
                 emb_ks,
                 emb_hs,
                 n_freqs,
                 hidden_channels,
                 n_head=4,
                 approx_qk_dim=512,
                 activation='prelu',
                 eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

        # Intra-Frame Full-Band Module
        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(in_channels,
                                 hidden_channels,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)
        self.intra_linear = nn.ConvTranspose1d(hidden_channels * 2,
                                               emb_dim,
                                               kernel_size=emb_ks,
                                               stride=emb_hs)

        # Sub-Band Temporal Module
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(in_channels,
                                 hidden_channels,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)
        self.inter_linear = nn.ConvTranspose1d(hidden_channels * 2,
                                               emb_dim,
                                               kernel_size=emb_ks,
                                               stride=emb_hs)

        # Cross-Frame Self-Attention Module
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0

        for ii in range(n_head):
            self.add_module(f"attn_conv_Q_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, E, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module(f"attn_conv_K_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, E, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module(f"attn_conv_V_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, emb_dim // n_head, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps)))
        self.add_module("attn_concat_proj",
                        nn.Sequential(nn.Conv2d(emb_dim, emb_dim, kernel_size=1),
                                      get_layer(activation)(),
                                      LayerNormalization4DCF((emb_dim, n_freqs), eps=eps)))

    def __getitem__(self, item):
        return getattr(self, item)

    def forward(self, x):
        B, C, old_T, old_F = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        F = math.ceil((old_F - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = nn.functional.pad(x, (0, F - old_F, 0, T - old_T))

        # Intra-Frame Full-Band Module
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, F]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, F)  # [BT, C, F]
        intra_rnn = nn.functional.unfold(intra_rnn[..., None],
                                         (self.emb_ks, 1),
                                         stride=(self.emb_hs, 1))  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn = self.intra_rnn(intra_rnn)[0]  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, F]
        intra_rnn = intra_rnn.view([B, T, C, F])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, F]
        intra_rnn = intra_rnn + input_  # [B, C, T, F]

        # Sub-Band Temporal Module
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * F, C, T)  # [BF, C, T]
        inter_rnn = nn.functional.unfold(inter_rnn[..., None],
                                         (self.emb_ks, 1),
                                         stride=(self.emb_hs, 1))  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn = self.inter_rnn(inter_rnn)[0]  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, F, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, F]
        inter_rnn = inter_rnn + input_  # [B, C, T, F]

        # Cross-Frame Self-Attention Module
        inter_rnn = inter_rnn[..., :old_T, :old_F]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self[f"attn_conv_Q_{ii}"](batch))  # [B, C, T, F]
            all_K.append(self[f"attn_conv_K_{ii}"](batch))  # [B, C, T, F]
            all_V.append(self[f"attn_conv_V_{ii}"](batch))  # [B, C, T, F/H]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, F]
        K = torch.cat(all_K, dim=0)  # [B', C, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C, T, F/H]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*F]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*F]
        V = V.transpose(1, 2)  # [B', T, C, F/H]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/H]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / emb_dim ** 0.5  # [B', T, T]
        attn_mat = nn.functional.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/H]

        V = V.reshape(old_shape)  # [B', T, C, F/H]
        V = V.transpose(1, 2)  # [B', C, T, F/H]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [H, B, C, T, F/H]
        batch = batch.transpose(0, 1)  # [B, H, C, T, F/H])
        batch = batch.contiguous().view([B, self.n_head * emb_dim, old_T, -1])  # [B, C, T, F]
        batch = self["attn_concat_proj"](batch)  # [B, C, T, F]
        out = batch + inter_rnn

        return out