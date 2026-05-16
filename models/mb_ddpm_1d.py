# 文件路径: alistion/lunwen3_gan/models/mb_ddpm_1d.py
import math
from typing import List
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F

# 编码正弦时间步嵌入
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        # Keep the lookup table on the same device as the model after .to(device).
        self.register_buffer("embeddings", embeddings)

    def forward(self, x, t):
        embeds = self.embeddings[t]
        return embeds[:, :, None]

# Residual Blocks
class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv1d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x

class Attention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        _, C, _ = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(float(C), device=x.device))
        attention_weight = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weight, v)
        h = self.proj(output)

        return x + h

class UnetLayer(nn.Module):
    def __init__(self, upscale: bool, attention: bool, num_groups: int, dropout_prob: float, C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose1d(C, C // 2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv1d(C, C * 2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x

class UNET_1D(nn.Module):
    def __init__(self,
                 Channels: List = [64, 128, 256, 512, 512, 384],
                 Attentions: List = [False, False, False, False, False, False],
                 Upscales: List = [False, False, False, True, True, True],
                 num_groups: int = 32,
                 dropout_prob: float = 0.1,
                 input_channels: int = 1,
                 output_channels: int = 1,
                 time_steps: int = 1000,
                 in_feature: int = 2048,
                 num_classes: int = 5):  # 默认修改为 2048，适配 lunwen3
        super().__init__()
        self.num_layers = len(Channels)
        target_feature = math.ceil(in_feature / 8) * 8
        if in_feature % 8 == 0:
            self.shallow_conv = nn.Conv1d(input_channels, Channels[0], kernel_size=3, padding=1)
        else:
            self.shallow_conv = nn.Conv1d(input_channels, Channels[0], kernel_size=target_feature-in_feature+1, padding=target_feature-in_feature)
        out_channels = (Channels[-1] // 2) + Channels[0]
        self.late_conv = nn.Conv1d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv1d(out_channels // 2, output_channels, kernel_size=target_feature-in_feature+1)
        self.relu = nn.ReLU(inplace=True)

        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        self.label_embedding = nn.Embedding(num_classes, max(Channels))

        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
            )
            setattr(self, f'Layer{i + 1}', layer)

    def forward(self, x, t, labels):
        x = self.shallow_conv(x)
        residuals = []
        embeddings = self.embeddings(x, t) + self.label_embedding(labels).unsqueeze(-1)
        for i in range(self.num_layers // 2):
            layer = getattr(self, f'Layer{i + 1}')
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f'Layer{i + 1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers - i - 1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))

class DDPM_Scheduler_1D(nn.Module):
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        beta = torch.linspace(1e-4, 0.02, num_time_steps)
        alpha = torch.cumprod(1 - beta, dim=0)
        # Register as buffers so scheduler.to(device) moves them together with
        # the module; otherwise CUDA timestep indices cannot index CPU tensors.
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)

    def forward(self, t):
        return self.beta[t], self.alpha[t]
