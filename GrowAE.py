'''
MODEL'S BASIC BLOCK
'''
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,\
                stride=1, kernel_size=3, padding=1)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,\
                     kernel_size=kernel_size, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,\
                     kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            if downsample is not None:
                self.downsample = downsample 
            else:
                self.downsample = nn.Sequential(
                   nn.Conv2d(in_channels, out_channels, kernel_size=3, \
                            stride=stride, padding=1, bias=False),
                   nn.BatchNorm2d(out_channels)
               )
        else:
            self.downsample = nn.Identity()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(x)
        # print(out.size(), residual.size())
        out += residual
        out = self.relu(out)
        return out


class FirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # input_size = (3,4,4)
        self.fromRGB = nn.Conv2d(3, 128, stride=1, kernel_size=1, padding=0) # Output: 128, 16, 16
        self.conv1 = ResidualBlock(128, 128, stride=2) # Output: 128, 8, 8
        self.conv2 = ResidualBlock(128, 128, stride=2) # Output: 128, 4, 4 

        self.old_encoder = nn.Identity()
        self.old_decoder = nn.Identity()

        self.deconv1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.toRGB = nn.Conv2d(128, 3, stride=1, kernel_size=1, padding=0)

    def forward(self, x, alpha):
        x = self.fromRGB(x)
        x = self.conv2(self.conv1(x))
        x = self.old_encoder(x)
        x = self.old_decoder(x)
        x = self.deconv2(self.deconv1(x))
        x = self.toRGB(x)
        return x

    def feature_extract(self, x):
        x = self.fromRGB(x)
        x = self.conv2(self.conv1(x))
        x = self.old_encoder(x)
        return x

    def reconstruct(self, x):
        x = self.old_decoder(x)
        x = self.deconv2(self.deconv1(x))
        x = self.toRGB(x)
        return x


# weighted sum output
class WeightedSum(nn.Module):
    # init with default value
    def __init__(self):
        super(WeightedSum, self).__init__()

    # output a weighted sum of inputs
    def forward(self, x, y, alpha):
        # only supports a weighted sum of two inputs
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - alpha) * x) + (alpha * y)
        return output

class GrowingModel(nn.Module):
    def __init__(self, model, n_layer):
        super().__init__()
        dimensions = [[128, 128], [128, 128], [128, 128], \
                    [64, 128], [32, 64], [16, 32], [8, 16]]
        dimension = dimensions[n_layer]

        self.fromRGB = nn.Conv2d(3, dimension[0], stride=1, kernel_size=1, padding=0)
        self.conv1 = ResidualBlock(dimension[0], dimension[1])  # increase dimension
        self.conv2 = ResidualBlock(dimension[1], dimension[1], stride=2) # downsample
        self.old_encoder = nn.Sequential(*list(model.children())[1:4])
        
        self.fromRGB_old = nn.Sequential(*list(model.children())[:1])
        self.downsample = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)

        
        self.upsample = nn.ConvTranspose2d(dimension[1], dimension[1], 4, 2, 1)
        self.toRGB_old = nn.Sequential(*list(model.children())[-1:])

        self.weight_sum = WeightedSum()
        self.relu = nn.ReLU(inplace=True)

        self.old_decoder = nn.Sequential(*list(model.children())[-4:-1])
        self.deconv1 = nn.ConvTranspose2d(dimension[1], dimension[1], 4, 2, 1) # upsample
        self.deconv2 = nn.Conv2d(dimension[1], dimension[0], stride=1, kernel_size=1, padding=0)
        self.toRGB = nn.Conv2d(dimension[0], 3, stride=1, kernel_size=1, padding=0)
        
        
    def forward(self, x, alpha = 0.0):
        new_encode = self.conv2(self.conv1(self.fromRGB(x)))
        old_encode = self.fromRGB_old(self.downsample(x))
        out_encode = self.old_encoder(self.weight_sum(old_encode, new_encode, alpha))

        old_decode = self.toRGB_old(self.upsample(self.old_decoder(out_encode)))
        decode = self.toRGB(self.deconv2(self.relu(self.deconv1(self.relu(self.old_decoder(out_encode))))))
        out = self.weight_sum(old_decode, decode, alpha)
        return out

    def feature_extract(self, x):
        x = self.fromRGB(x)
        x = self.conv2(self.conv1(x))
        x = self.old_encoder(x)
        return x

    def reconstruct(self, x):
        x = self.relu(self.old_decoder(x))
        x = self.relu(self.deconv1(x))
        x = self.deconv2(x)
        x = self.toRGB(x)
        return x

def init_model(n_layers):
    model = FirstLayer()
    for i in range(1, n_layers):
        model = GrowingModel(model, i)
    return model