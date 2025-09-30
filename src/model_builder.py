import torch
import torch.nn as nn
import torch.nn.functional as F

class Model1(nn.Module):
    def __init__(self, dropout_value=0.01):
        super(Model1, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False, padding=1, dilation=2), # Used Dilated kernels here instead of MP or strided convolution
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value))
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value))
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value))
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False, padding=1, groups=32),  # Used Depthwise separable convolution
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))  # Used GAP
        self.block5 = nn.Sequential(
          nn.Linear(64, 128), # Used FC
          nn.Linear(128, 256),
          nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view((x.shape[0],-1))
        x = self.block5(x)
        x = F.log_softmax(x, dim=1)
        return x
