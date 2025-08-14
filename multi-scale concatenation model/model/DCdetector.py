import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import DAC_structure, AttentionLayer
from .embed import DataEmbedding
from .RevIN import RevIN
from itertools import chain
from tkinter import _flatten


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list


class DCdetector(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=[3, 5, 7], channel=55,
                 d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(DCdetector, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size

        # Patching List
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size // patchsize, d_model, dropout))

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(win_size, patch_size, channel, False, attention_dropout=dropout,
                                  output_attention=output_attention),
                    d_model, patch_size, channel, n_heads, win_size) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x, pool_sizes):
        B, L, M = x.shape
        #print("L",L)
        revin_layer = RevIN(num_features=M)

        # 实例归一化
        x = revin_layer(x, 'norm')
        x_ori = self.embedding_window_size(x)

        total_patch_num = []
        total_patch_size = []

        series_patch_mean = []
        prior_patch_mean = []

        for pool_size in pool_sizes:
            if L % pool_size == 0:
                # 池化操作
                x_pooled = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=pool_size, stride=pool_size)
                x_pooled = x_pooled.permute(0, 2, 1)
                # 对池化后的数据进行上采样，恢复原始长度 L
                x_pooled = F.interpolate(x_pooled.permute(0, 2, 1), size=L, mode='linear', align_corners=True)
                x_pooled = x_pooled.permute(0, 2, 1)  # 恢复形状 (B, L, M)

                # 对池化后的数据进行多尺度划分
                for patch_index, patchsize in enumerate(self.patch_size):
                    x_patch_size, x_patch_num = x_pooled, x_pooled
                    x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')
                    x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')

                    x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)
                    x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)

                    x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p=patchsize)
                    x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)

                    '''print("x_patch_size shape", x_patch_size.shape)
                    print("x_patch_num shape", x_patch_num.shape)'''

                    total_patch_size.append(x_patch_size)
                    total_patch_num.append(x_patch_num)
                    # 编码器检测
                    series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)

                    '''print("series shape", [item.shape for item in series])
                    print("prior shape", [item.shape for item in prior])'''

                    series_patch_mean.append(series)
                    prior_patch_mean.append(prior)

        # 拼接 total_patch_size 和 total_patch_num
        if total_patch_size and total_patch_num:  # 确保列表不为空
            total_patch_size = torch.cat(total_patch_size, dim=1)  # 沿num_patches维度拼接
            total_patch_num = torch.cat(total_patch_num, dim=1)  # 沿num_patches维度拼接
        else:
            total_patch_size = total_patch_num = None

        '''print("total_patch_size shape", total_patch_size.shape)
        print("total_patch_num shape", total_patch_num.shape)'''

        super_series, super_prior = self.encoder(total_patch_size, total_patch_num, x_ori, -1)

        '''print("super_series shape", [item.shape for item in super_series])
        print("super_prior shape", [item.shape for item in super_prior])'''

        series_patch_mean.append(super_series)
        prior_patch_mean.append(super_prior)
        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))

        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        else:
            return None