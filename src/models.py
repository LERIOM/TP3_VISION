import torch.nn as nn

IMG_SIZE = 256

class FastFoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1) # 256x256 -> 128x128
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) # 128x128 -> 64x64
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1) # 64x64 -> 32x32
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # 32x32 -> 16x16
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1) # 16x16 -> 8x8
        self.act5 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, num_classes)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.act(out)
        return out
    
class ResNetLikeClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.block1 = ConvBlock(64, 64)
        self.block1_b = ConvBlock(64, 128, stride=2)

        self.block2 = ConvBlock(128, 128)
        self.block2_b = ConvBlock(128, 256, stride=2)
        
        self.block3 = ConvBlock(256, 256)
        self.block3_b = ConvBlock(256, 512, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.block1(x)
        x = self.block1_b(x)
        x = self.block2(x)
        x = self.block2_b(x)
        x = self.block3(x)
        x = self.block3_b(x)

        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x