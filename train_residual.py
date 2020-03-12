'''
LOAD LIBRARIES AND DATA
'''
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

img_size = (1024, 1024)
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
epochs = 50
data_dir = 'dataset/train/'
dataset = datasets.ImageFolder(data_dir,
                               data_transforms['train'])
loader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                             shuffle=True, num_workers=2)
dataset_sizes = len(dataset)
print("Dataset size: ", dataset_sizes)
print("Number of epochs: ", epochs)


'''
START TRAINING THE MODEL
'''

start = time.time()

model = MyModel()
print(str(model))

funcs = {
    'BCE' : nn.BCEWithLogitsLoss(),
    'MSE' : nn.MSELoss(),
    'VGG' : nn.MSELoss(),
    'MAE' : nn.L1Loss(),
}
loss_func = 'BCE'
criterion = funcs[loss_func]
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# layer_name = 'encoder'
# model.encoder.register_forward_hook(get_activation(layer_name))

loss_hist = []

print ("Start training model with {} images, size {}, in {} epochs, loss function {} ..."\
       .format(dataset_sizes, img_size, epochs, loss_func))
for epoch in range(epochs):
    print("Epoch {} ...".format(epoch))
    loss_per_epoch = 0
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        
        loss_per_epoch += loss.item() * data.size(0) 
        print('Batch idx {}, loss {}'.format(batch_idx, loss.item()))
    loss_hist.append(loss_per_epoch/dataset_sizes)
end = time.time() - start
print("Finish in {:.0f}h:{:.0f}m:{:.0f}s".format(end//3600, end//60, end%60))

# save the model 
torch.save(model.state_dict(), "results/models/encoder_{}_{}_{}.pth".format(img_size, epochs, loss_func))

# plot loss history
plt.plot(loss_hist)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('results/img/loss_{}_{}.png'.format(img_size, loss_func))

'''
VISUALIZE SOME RESULTS
'''
import math

def normalize_output(img):
    # img = img - img.min()
    # img = img / img.max()
    img = img * 255.
    return img
# data, _ = loader[0]
# Plot random input/output of model
idx = torch.randint(0, output.size(0), ())
pred = normalize_output(output[idx, 0])
img = data[idx, 0]
# print(img.size())
fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(img.detach().numpy(), cmap='gray')
axarr[0].title.set_text('Input')
axarr[0].axis('off')
axarr[1].imshow(pred.detach().numpy(), cmap='gray')
axarr[1].title.set_text('Output')
axarr[1].axis('off')
plt.savefig('results/img/reconstruct_{}_{}.png'.format(img_size, loss_func))


# Visualize feature maps
# tmp_model = model.copy()
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation = {}

layer_name = 'encoder'
model.encoder.register_forward_hook(get_activation(layer_name))

idx = torch.randint(0, dataset_sizes, ())
data, target = dataset[idx]
data.unsqueeze_(0)
output = model(data)

act = activation[layer_name].squeeze()
print("Feature size: ", act.size())
num_layers = act.size(0) 
ncols = 6
nrows = math.ceil(num_layers/ncols)

fig, axs = plt.subplots(nrows, ncols, figsize=(15,15))
idx = 0
for idx in range(num_layers):
    x = idx // ncols
    y = idx % ncols
    axs[x, y].imshow(act[idx], cmap='gray')
    axs[x, y].axis('off')
    axs[x, y].title.set_text('Layer {}'.format(idx))
#     idx += 1

# plt.margins(0, 0)
plt.savefig('results/img/layer_{}_{}_{}.png'.format(layer_name, img_size, loss_func))