import torch.nn as nn

class RadarPoseNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8))
        )

        self.fc = nn.Sequential(
            nn.Linear(32*8*8,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,34)
        )


    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)
