import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange,repeat
from timm.models.layers import DropPath
import torch.nn.functional as F
from mamba_ssm import Mamba

from config import config


class ChannelAttention1(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return self.fc(y).view(b, c, 1, 1)

class SpatialAttention1(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 压缩通道维度获取空间权重
        avg_result = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(avg_result)

# --- Mamba 相关模块 ---

class BMblock_spe(nn.Module):
    """光谱双向Mamba模块 (针对1D/组序列)"""
    def __init__(self, embed_dim, depth=1):
        super().__init__()
        self.mamba1 = nn.ModuleList([Mamba(embed_dim) for _ in range(depth)])
        self.mamba2 = nn.ModuleList([Mamba(embed_dim) for _ in range(depth)])
        
        # 1D 卷积序列
        self.pointconv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, 1),
                nn.GroupNorm(1, embed_dim)
            ) for _ in range(3)
        ])
        
        self.drop_path = DropPath(0.0)

    def forward(self, x):
        # x shape: (B, L, C) -> (64, 4, 16)
        identity = x
        
        # 正向路径
        x1 = self.pointconv[0](x.permute(0, 2, 1)).permute(0, 2, 1)
        for mamba in self.mamba1:
            x1 = mamba(x1)
        x1 = self.pointconv[1](x1.permute(0, 2, 1)).permute(0, 2, 1)

        # 反向路径
        x2 = torch.flip(identity, dims=[1])
        for mamba in self.mamba2:
            x2 = mamba(x2)
        x2 = torch.flip(x2, dims=[1])
        x2 = self.pointconv[2](x2.permute(0, 2, 1)).permute(0, 2, 1)

        return self.drop_path(x1 + x2) + identity

class BMblock(nn.Module):
    """空间双向Mamba模块 (针对2D特征图展平序列)"""
    def __init__(self, embed_dim, depth=1):
        super().__init__()
        self.mamba1 = nn.ModuleList([Mamba(embed_dim) for _ in range(depth)])
        self.mamba2 = nn.ModuleList([Mamba(embed_dim) for _ in range(depth)])
        
        # 空间投影卷积组
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            ) for _ in range(3)
        ])
        
        self.drop_path = DropPath(0.0)

    def forward(self, x):
        # x shape: (B, C, L) -> (64, 64, 169)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        identity = x

        # 1. 初始空间变换
        x_in = self.convs[0](x.view(B, C, H, W)).view(B, C, L).permute(0, 2, 1)

        # 2. 正向 Mamba
        x1 = x_in
        for mamba in self.mamba1:
            x1 = mamba(x1)
        x1 = self.convs[1](x1.permute(0, 2, 1).view(B, C, H, W)).view(B, C, L)

        # 3. 反向 Mamba
        x2 = torch.flip(x_in, dims=[1])
        for mamba in self.mamba2:
            x2 = mamba(x2)
        x2 = torch.flip(x2, dims=[1])
        x2 = self.convs[2](x2.permute(0, 2, 1).view(B, C, H, W)).view(B, C, L)

        # 4. 融合与残差
        return self.drop_path((x1 + x2) / 2) + identity


class Net(nn.Module):
    def __init__(self, num_band=103, embed_dim=64, num_class=10, drop_path=0.0, bi=True, fu=True):
        super().__init__()

        self.name = config.dataset
        self.depth = (config.num_band - 9) // 5 + 1
        self.head_c = 48
        
        # --- 基础特征提取层 ---
        self.conv_head = nn.Sequential(
            nn.Conv2d(config.num_band, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.1),
            nn.GELU()
        )

        # --- 光谱特征分支 (Spectral Branch) ---
        self.spe_conv1 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 3), padding=(0, 0, 1)),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.atten_spe = ChannelAttention1(64)
        
        # 光谱 Mamba 模块
        self.biMamba_spe1 = BMblock_spe(embed_dim=16)
        self.biMamba_spe2 = BMblock_spe(embed_dim=16)
        self.biMamba_spe3 = BMblock_spe(embed_dim=16)
        self.biMamba_spe4 = BMblock_spe(embed_dim=16)

        # --- 空间特征分支 (Spatial Branch) ---
        self.conv2ds_spa1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.atten_spa = SpatialAttention1()

        # 空间多尺度卷积组
        dim = 64
        self.conv_w11 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv_w12 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv_w13 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        
        self.conv_w21 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv_w22 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv_w23 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)

        self.conv_h11 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.conv_h12 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.conv_h13 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv_h21 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.conv_h22 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.conv_h23 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        # 空间 Mamba 模块
        self.biMamba1 = BMblock(embed_dim, bi)
        self.biMamba2 = BMblock(embed_dim, bi)
        self.biMamba3 = BMblock(embed_dim, bi)
        self.biMamba4 = BMblock(embed_dim, bi)

        # --- 分类与融合层 ---
        self.weights_cla = nn.Parameter(torch.ones(2))
        self.softmax_cla = nn.Softmax(dim=0)
        self.softmax_cla_spe = nn.Softmax(dim=-1)

        self.avg_2d = nn.AdaptiveAvgPool2d(1)
        self.mlp_head = nn.Linear(embed_dim, num_class)
        self.fc_cla1d_spe = nn.Linear(embed_dim, num_class)

    # --- 辅助变换函数 ---
    def data_transformW(self, x):
        B, C, H, W = x.shape
        mid = W // 2
        output = torch.zeros((B, C, H + 1, W), dtype=x.dtype, device=x.device)
        row_indices = torch.arange(H, device=x.device)
        output[:, :, row_indices, mid + 1:] = x[:, :, row_indices, :mid]
        output[:, :, row_indices + 1, :mid + 1] = x[:, :, row_indices, mid:]
        return output

    def data_transformH(self, x):
        B, C, H, W = x.shape
        mid = H // 2
        output = torch.zeros((B, C, H, W + 1), dtype=x.dtype, device=x.device)
        col_indices = torch.arange(W, device=x.device)
        output[:, :, mid + 1:, col_indices] = x[:, :, :mid, col_indices]
        output[:, :, :mid + 1, col_indices + 1] = x[:, :, mid:, col_indices]
        return output

    def group_offset(self, data, offset):
        B, _ = data.shape
        result = torch.roll(data, shifts=-offset, dims=1)
        return result.view(B, 4, -1)

    def symmetric_avg_pool2d(self, x, kernel_size=3):
        pad = kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        return F.avg_pool2d(x_padded, kernel_size=kernel_size, stride=1)

    def average_adjacent_channels(self, data):
        B, C, H, W = data.shape
        padded = F.pad(data, (0, 0, 0, 0, 1, 1), mode='reflect')
        # 矢量化计算相邻通道平均
        return (padded[:, :-2, :, :] + padded[:, 1:-1, :, :] + padded[:, 2:, :, :]) / 3

    # --- 前向传播 ---
    def forward(self, x):
        # 1. 初始输入转换 (B, S, H, W)
        x1 = x.permute(0, 3, 1, 2)
        x1_spa = self.conv_head(x1)
        B, C, H, W = x1_spa.shape

        # 2. 光谱分支处理
        x1_spe = x1_spa.permute(0, 2, 3, 1).unsqueeze(1) # (B, 1, H, W, C)
        x1_spe_conv = self.spe_conv1(x1_spe).squeeze(1).permute(0, 3, 1, 2)
        
        # 光谱注意力与池化
        x1_spe_en = x1_spe_conv * self.atten_spe(self.average_adjacent_channels(x1_spe_conv))
        x1_spe_pool = F.adaptive_avg_pool2d(x1_spe_en, 1).view(B, -1)

        # 光谱组 Mamba (Group Offset Mamba)
        offsets = [0, 4, 8, 12]
        mamba_spe_outputs = []
        for i, off in enumerate(offsets):
            mamba_mod = getattr(self, f'biMamba_spe{i+1}')
            m_out = mamba_mod(self.group_offset(x1_spe_pool, off)).flatten(1)
            mamba_spe_outputs.append(torch.roll(m_out, shifts=off, dims=1))
        
        x1_spe_mamba_final = sum(mamba_spe_outputs) / 4

        # 3. 空间分支处理
        x_spa_en = self.conv2ds_spa1(x1_spa) * self.atten_spa(self.symmetric_avg_pool2d(self.conv2ds_spa1(x1_spa)))
        
        # 错位变换
        x_tran_w = self.data_transformW(x_spa_en)
        x_tran_h = self.data_transformH(x_spa_en)

        # W-方向多路径 Mamba
        w_fuse = (self.conv_w11(x_spa_en) + self.conv_w12(x_spa_en) + self.conv_w13(x_spa_en)) / 3 + x_spa_en
        x1_w_mamba = self.biMamba1(w_fuse.flatten(2)).view(B, C, H, W)

        # W-错位 Mamba
        mid_w = W // 2
        tw_fuse = (self.conv_w21(x_tran_w) + self.conv_w22(x_tran_w) + self.conv_w23(x_tran_w)) / 3 + x_tran_w
        tw_flat = tw_fuse.flatten(2)[:, :, (mid_w+1) : ((H+1)*W - mid_w)]
        x1_tw_mamba = self.biMamba2(tw_flat).view(B, C, H, W)

        # H-方向多路径 Mamba
        h_fuse = (self.conv_h11(x_spa_en) + self.conv_h12(x_spa_en) + self.conv_h13(x_spa_en)) / 3 + x_spa_en
        x1_h_mamba = self.biMamba3(h_fuse.permute(0, 1, 3, 2).flatten(2)).view(B, C, W, H).permute(0, 1, 3, 2)

        # H-错位 Mamba
        mid_h = H // 2
        th_fuse = (self.conv_h21(x_tran_h) + self.conv_h22(x_tran_h) + self.conv_h23(x_tran_h)) / 3 + x_tran_h
        th_flat = th_fuse.permute(0, 1, 3, 2).flatten(2)[:, :, (mid_h+1) : (H*(W+1) - mid_h)]
        x1_th_mamba = self.biMamba4(th_flat).view(B, C, W, H).permute(0, 1, 3, 2)

        # 空间融合
        x_spa_mamba_final = self.mlp_head(self.avg_2d((x1_w_mamba + x1_tw_mamba + x1_h_mamba + x1_th_mamba) / 4).view(B, -1))

        # 4. 最终分类决策融合
        x1_spe_logits = self.softmax_cla_spe(self.fc_cla1d_spe(x1_spe_mamba_final))
        w_cla = self.softmax_cla(self.weights_cla)
        
        return x_spa_mamba_final * w_cla[0] + x1_spe_logits * w_cla[1]
    
    
    
  
    
   