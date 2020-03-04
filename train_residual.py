  # extract feature from Water Crystal Dataset 
import argparse
import os
import numpy as np
import math
from matplotlib import pyplot as plt

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

import time

from model import MyModel

img_size = (32, 32)
# data transform
data_transforms = {
    'train': transforms.Compose([
        # transforms.Grayscale(),
        # transforms.RandomResizedCrop(512),
        transforms.Resize(img_size),
        # transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [255., 255., 255.])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize(1024),
        # transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [255., 255., 255.])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
epochs = 10
data_dir = 'dataset/train/'
dataset = datasets.ImageFolder(data_dir,
                               data_transforms['train'])
loader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                             shuffle=True, num_workers=2)
dataset_sizes = len(dataset)
print("Dataset size: ", dataset_sizes)

start = time.time()

model = MyModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_hist = []
for epoch in range(epochs):
    loss_per_epoch = 0
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        
        loss_per_epoch += loss.item() * data.size(0) 
        print('Epoch {}, Batch idx {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
    loss_hist.append(loss_per_epoch/dataset_sizes)
end = time.time() - start
print("Finish in {:.0f}m:{:.0f}s".format(end//60, end%60))

def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    # img = img * 255.
    return img

# Plot some images
idx = torch.randint(0, output.size(0), ())
pred = normalize_output(output[idx, 0])
img = data[idx, 0]

# plot loss history
plt.plot(loss_hist)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('autoencoder_{}_loss.png'.format(img_size))


# Visualize feature maps
# tmp_model = model.copy()
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation = {}

layer_name = 'residual'
model.residual.register_forward_hook(get_activation(layer_name))

data, _ = dataset[0]
data.unsqueeze_(0)
output = model(data)

act = activation[layer_name].squeeze()
num_layers = act.size(0) 
fig, axarr = plt.subplots(1, num_layers, figsize=(20,20))
for idx in range(num_layers):
    axarr[idx].imshow(act[idx], cmap='gray')
    axarr[idx].axis('off')
    axarr[idx].title.set_text('Layer {}'.format(idx))

plt.savefig('layer_{}_{}.png'.format(layer_name, img_size))

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(data.detach().numpy(), cmap='gray')
axarr[0].title.set_text('Input')
axarr[0].axis('off')
axarr[1].imshow(output.detach().numpy(), cmap='gray')
axarr[1].title.set_text('Output')
axarr[1].axis('off')
plt.savefig('autoencoder_{}.png'.format(img_size))