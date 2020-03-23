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
from dataset import UnlabelDataset

start = time.time()

img_size = (1024, 1024)
# data transform
data_transforms = {
    'train': transforms.Compose([
        # transforms.Grayscale(),
        # transforms.RandomResizedCrop(512),
        transforms.Resize(img_size),
        # transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize(1024),
        # transforms.CenterCrop(512),
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset/train/500'
# dataset = datasets.ImageFolder(data_dir,
#                                data_transforms['train'])
batch_size = 8
dataset = UnlabelDataset(data_dir, data_transforms['train'])
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=8)
dataset_sizes = len(dataset)
end = time.time() - start
print("Load and crop take {:.0f}m for {} images".format(end//60, dataset_sizes))
print("Dataset size: ", dataset_sizes)
print("Number of batchs per epoch: ", dataset_sizes//batch_size)


'''
START TRAINING THE MODEL
'''
# release GPU before running
# del model, data
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MyModel()
model.to(device)

start = time.time()
print(str(model))

rae_version = 'v1'
funcs = {
    'BCE' : nn.BCEWithLogitsLoss(),
    'MSE' : nn.MSELoss(),
    'VGG' : nn.MSELoss(),
    'MAE' : nn.L1Loss(),
}
loss_func = 'BCE'
lr = 1e-3
epochs = 100

criterion = funcs[loss_func]
# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_hist = []

print ("Start training model with {} images, size {}, lr = {}, in {} epochs, loss function {}"\
       .format(dataset_sizes, img_size, lr, epochs, loss_func))
for epoch in range(epochs):
    print("Epoch {} ...".format(epoch))
    loss_per_epoch = 0
    epoch_start = time.time()
#     for batch_idx, (data, target) in enumerate(loader):
    for batch_idx, (data, filenames) in enumerate(loader):        
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
#         pred_features = VGG(output)
#         true_features = VGG(data)
#         content_loss = MSE(pred_features.relu4_3, true_features.relu4_3)
#         mse_loss = MSE(output, data)
#         loss = 0.3 * content_loss + 0.7 * mse_loss
        loss.backward()
        optimizer.step()
        
        loss_per_epoch += loss.item() * data.size(0) 
        epoch_end = time.time() - epoch_start
        epoch_time = "{:.0f}m:{:.0f}s".format(epoch_end//60, epoch_end%60)
        print('Epoch {}, batch idx {}, loss {}, take {}'.format(epoch, batch_idx, loss.item(), epoch_time))
    loss_hist.append(loss_per_epoch/dataset_sizes)
end = time.time() - start
print("Finish in {:.0f}h:{:.0f}m:{:.0f}s".format(end//3600, (end//60)%60, end%60))

# save the model 
model_name = "results/models/rae_{}_{}_{}.pth".format(rae_version, dataset_sizes, epochs)
torch.save(model.state_dict(), model_name)

# plot loss history
plt.plot(loss_hist)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.savefig('results/img/loss_{}_{}_{}.png'.format(rae_version, dataset_sizes, rae_version))

'''
VISUALIZE SOME RESULTS
Follow the guide from this post: https://discuss.pytorch.org/t/visualize-feature-map/29597/7
'''
import math

# save_model = MyModel()
# sdict = torch.load(model_name)
# save_model.load_state_dict(sdict)

def normalize_output(img):
    # img = img - img.min()
    # img = img / img.max()
#     img = img * 255. + 127.5
    return img


# idx = torch.randint(0, output.size(0), ())
idx = 2
pred = normalize_output(output[idx, 0])
img = data[idx, 0]
img_name = filenames[idx]

print(idx)
fig, axarr = plt.subplots(1, 2, figsize=(20, 20))
axarr[0].imshow(img.cpu().detach().numpy(), cmap='gray')
axarr[0].title.set_text('Input')
axarr[0].axis('off')
axarr[1].imshow(pred.cpu().detach().numpy(), cmap='gray')
axarr[1].title.set_text('Output')
axarr[1].axis('off')
plt.savefig('results/img/{}_{}_{}_{}_RAEresconstruct.png'.format(img_name, rae_version, dataset_sizes, epochs))


# Visualize feature maps
# tmp_model = model.copy()
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation = {}
layer_name = 'encoder'
model.encoder.register_forward_hook(get_activation(layer_name))

img = data[idx]
pred = model(img.unsqueeze(0).to(device))

act = activation[layer_name].squeeze().cpu()
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

plt.savefig('results/img/{}_{}_{}_{}_RAElayer.png'.format(img_name, rae_version, dataset_sizes, epochs))