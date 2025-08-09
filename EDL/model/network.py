 import torch
import torch.nn as nn
from .blocks import ConvBNAct

class TinyBackbone(nn.Module):
    def __init__(self, c1=3, c2=128):
        super().__init__()
        self.stem = ConvBNAct(c1, 32, 3, 2)
        self.c2 = ConvBNAct(32, 64, 3, 2)
        self.c3 = nn.Sequential(
            ConvBNAct(64, 64, 3, 1),
            ConvBNAct(64, 64, 3, 1),
            ConvBNAct(64, 96, 1, 1)
        )
        self.c4 = ConvBNAct(96, 128, 3, 2)
        self.c5 = nn.Sequential(
            ConvBNAct(128, 128, 3, 1),
            ConvBNAct(128, 128, 3, 1),
            ConvBNAct(128, c2, 1, 1)
        )
        self.c6 = ConvBNAct(c2, c2, 3, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        return x

class SimpleNeck(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.m = nn.Sequential(
            ConvBNAct(c, c, 3, 1),
            ConvBNAct(c, c, 3, 1)
        )

    def forward(self, x):
        return self.m(x)

class Head(nn.Module):
    def __init__(self, c=128, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.out_ch = 1 + num_classes + 4
        self.pred = nn.Conv2d(c, self.out_ch, 1, 1, 0)

    def forward(self, x):
        return self.pred(x)

class MiniYOLO(nn.Module):
    def __init__(self, num_classes: int, channels: int = 128):
        super().__init__()
        self.backbone = TinyBackbone(3, channels)
        self.neck = SimpleNeck(channels)
        self.head = Head(channels, num_classes)
        self.stride = 16
        self.num_classes = num_classes

    def forward(self, x):
        f = self.backbone(x)
        f = self.neck(f)
        p = self.head(f)
        return p
