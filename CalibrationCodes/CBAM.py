import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        attn_map = self.conv1(x_cat)
        return x * self.sigmoid(attn_map)
# ===============================

class Siamese(nn.Module):
    def __init__(self, model_channels=64, in_channels=1, img_size=(176, 112)):
        super().__init__()
        H0, W0 = img_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, model_channels, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=8, num_channels=model_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) 
        )

        ch = model_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(ch, 2*ch, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=2 * ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) 
        )
        ch *= 2

        self.spatial_attn = SpatialAttention(kernel_size=7)

        self.block2 = nn.Sequential(
            nn.Conv2d(ch, 2*ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=2 * ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*ch, 2*ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) 
        )
        ch *= 2
        
        self.block3 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((3, 2))
        flatten_dim = ch * 3 * 2  

        # 3) MLP head
        self.mlp = nn.Sequential(
            nn.Linear(flatten_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LayerNorm(512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward_one(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.spatial_attn(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

    def forward(self, x1, x2):
        h1 = self.forward_one(x1)
        h2 = self.forward_one(x2)
        d = torch.cat([torch.abs(h1-h2), h1*h2], dim=1)
        out = self.classifier(d)
        return out
