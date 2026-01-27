"""
Visual Context Adapter for Flux Model

本模块实现 Visual Context Adapter，用于将视觉骨架图的空间约束信息注入 Flux.1 模型。
包含:
1. ConvEncoder: 提取骨架图多尺度特征
2. VisualContextAdapter: 适配器主类，包含编码器和零初始化投影层

参考文献:
- ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models
- T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, List, Tuple

class ResnetBlock(nn.Module):
    """
    基础残差卷积块
    """
    def __init__(self, in_channels: int, out_channels: int = None, stride: int = 1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class ConvEncoder(nn.Module):
    """
    骨架图编码器
    
    结构: 4个残差卷积块，逐步提取特征并下采样
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        base_channels: int = 64,
        out_channels: int = 3072,  # Flux hidden size (e.g. 3072 for dev)
        num_blocks: int = 4,
        downsample_factor: int = 8 # Target downsample ratio (e.g., 8 to match VAE)
    ):
        super().__init__()
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        
        # 确定哪些层需要下采样
        # 如果需要下采样8倍，可以在前3层进行stride=2
        downsamples_needed = 0
        if downsample_factor > 1:
            import math
            downsamples_needed = int(math.log2(downsample_factor))
        
        for i in range(num_blocks):
            # 增加通道数
            next_channels = min(current_channels * 2, 512) 
            # 决定是否下采样
            stride = 2 if i < downsamples_needed else 1
            
            self.blocks.append(
                ResnetBlock(current_channels, next_channels, stride=stride)
            )
            current_channels = next_channels
            
        # 最终投影到目标维度
        self.final_proj = nn.Conv2d(current_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x: [B, 3, H, W]
        h = self.conv_in(x)
        
        for block in self.blocks:
            h = block(h)
            
        h = self.final_proj(h)
        return h # [B, out_channels, h, w]


class VisualContextAdapter(nn.Module):
    """
    Visual Context Adapter
    
    功能:
    1. 接收视觉骨架图 (H, W, 3)
    2. 编码为特征图 (h, w, d)
    3. 提供零初始化的可学习缩放因子 alpha (通常在注入点使用，这里也可以提供特征)
    
    Design Note:
    根据公式 H_img' = H_img + alpha * CrossAttn(H_img, F_skel)
    本模块主要负责计算 F_skel。
    alpha作为参数可以存储在这里，也可以存储在DoubleStreamBlock中。
    为了方便管理，这里提供特征提取功能。
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 3072, # Match Flux hidden size
        downsample_factor: int = 8
    ):
        super().__init__()
        self.encoder = ConvEncoder(
            in_channels=in_channels, 
            out_channels=hidden_size,
            downsample_factor=downsample_factor
        )
        
        # 零初始化线性层 (用于特征注入前的最后处理，如果需要)
        # 实际上，如果是在CrossAttention中使用，通常是作为KV
        # 这里我们保留一个可选的Projection，如果不需要可以恒等
        self.feature_proj = nn.Identity()
        
        # alpha 参数通常与注入点绑定，但如果需要集中管理
        # 这里定义一个全局alpha示例，或者在每个block中定义
        # 这里我们只负责输出 F_skel
        
    def forward(self, x):
        """
        Args:
            x: 骨架图 [B, 3, H, W]
            
        Returns:
            features: [B, L, D] or [B, D, h, w] depending on usage
            Flux usually expects sequences [B, L, D]
        """
        # Encoder output: [B, D, h, w]
        features = self.encoder(x)
        
        # Flatten to sequence [B, D, L] -> [B, L, D]
        # h*w = L
        b, c, h, w = features.shape
        features = rearrange(features, 'b c h w -> b (h w) c')
        
        return features

def zero_module(module):
    """
    Zero initialize the weights of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
