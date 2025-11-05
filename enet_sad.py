import torch
import torch.nn as nn
import torch.nn.functional as F

# ENet_Block

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels-3, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(x)
        x = torch.cat([conv, pool], dim=1)
        x = self.bn(x)
        return self.prelu(x)


# ENet_Bottleneck

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dropout_prob=0.1):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        internal = out_channels // 4

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, internal, kernel_size=2, stride=2, padding=0, bias=False)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, internal, kernel_size=1, bias=False)
            self.shortcut = None

        self.bn1 = nn.BatchNorm2d(internal)
        self.conv2 = nn.Conv2d(internal, internal, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal)
        self.conv3 = nn.Conv2d(internal, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout2d(dropout_prob)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        if self.downsample:
            identity = self.shortcut(identity)

        return self.prelu(out + identity)

# ENet-SAD
class ENet_SAD(nn.Module):
    def __init__(self, backbone='enet', sad=True, num_classes=2):
        super(ENet_SAD, self).__init__()
        self.initial = InitialBlock(3, 16)
        self.down1 = Bottleneck(16, 64, downsample=True)
        self.down2 = Bottleneck(64, 128, downsample=True)

        # Encoder_residual_blocks
        self.bottlenecks = nn.Sequential(
            Bottleneck(128, 128),
            Bottleneck(128, 128),
            Bottleneck(128, 128)
        )

        # SAD head (attention)
        self.sad = sad
        if sad:
            self.attention = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=1),
                nn.Sigmoid()
            )

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.out_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.bottlenecks(x)

        if self.sad:
            attn = self.attention(x)
            x = x * attn

        x = self.up1(x)
        x = self.up2(x)
        x = self.out_conv(x)
        return x
