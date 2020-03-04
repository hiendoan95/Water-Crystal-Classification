import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
# from data_utils import TensorDataset
# data_utils.TensorDataset

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.pool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(in_channels, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, out_channels, 3, 1, 1)
        # self.conv3 = nn.Conv2d(12, 24, 3, 1, 1)
        # self.conv4 = nn.Conv2d(24, out_channels, 3, 1, 1)
        
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x))) 
        # x = F.relu(self.pool(self.conv3(x)))        
        # x = F.relu(self.pool(self.conv4(x))) 
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # decoder
        # self.conv_trans4 = nn.ConvTranspose2d(in_channels, 24, 4, 2, 1)
        # self.conv_trans3 = nn.ConvTranspose2d(24, out_channels, 4, 2, 1)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels, 6, 4, 2, 1)
        self.conv_trans1 = nn.ConvTranspose2d(6, out_channels, 4, 2, 1)
        # self.conv_trans2 = nn.ConvTranspose2d(6, 3, 4, 2, 1)
        
    def forward(self, x):
        # x = F.relu(self.conv_trans4(x))
        # x = F.relu(self.conv_trans3(x))
        x = F.relu(self.conv_trans2(x))
        x = self.conv_trans1(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
        #                   padding=0, dilation=1, groups=1, bias=True, 
        #                   padding_mode='zeros')
        super(MyModel, self).__init__()

        # encoder
        # self.conv = nn.Conv2d(1, )
        self.encoder = EncoderBlock(in_channels=3, out_channels=12)
        self.residual = ResidualBlock(in_channels=12, out_channels=12)
        self.decoder = DecoderBlock(in_channels=12, out_channels=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x