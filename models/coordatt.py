import torch
import torch.nn as nn

class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, W]
        mid_channels = max(1, in_channels // reduction)

        # Réduction dimensionnelle
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)

        # Ré-expansion vers out_channels
        self.conv_h = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # Si les dimensions ne matchent pas, projection résiduelle
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.size()

        # Pooling moyen
        zh = x.mean(dim=3)  # [B, C, H]
        zw = x.mean(dim=2)  # [B, C, W]

        # Concaténation spatiale
        cat = torch.cat([zh, zw], dim=2)  # [B, C, H+W]

        # Réduction
        y = self.conv1(cat)  # [B, mid_channels, H+W]

        # Séparation H et W
        fh, fw = torch.split(y, [H, W], dim=2)  # [B, mid_channels, H], [B, mid_channels, W]

        # Ré-expansion + Sigmoid
        gh = self.sigmoid(self.conv_h(fh)).unsqueeze(3)  # [B, out_channels, H, 1]
        gw = self.sigmoid(self.conv_w(fw)).unsqueeze(2)  # [B, out_channels, 1, W]

        # Si besoin, adapter x pour le résidu
        x_proj = self.shortcut(x)

        # Application attention
        out = x_proj * gh * gw
        return out
