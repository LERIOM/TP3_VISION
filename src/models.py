import torch.nn as nn

IMG_SIZE = 128

class FastFoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1) # 128x128 -> 64x64
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) # 64x64 -> 32x32
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1) # 32x32 -> 16x16
        self.act3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.act4 = nn.ReLU() 
        self.fc2 = nn.Linear(512, num_classes)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x