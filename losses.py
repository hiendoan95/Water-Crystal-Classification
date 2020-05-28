import torch
import torch.nn as nn

def Spherical(x, y):
    cos = nn.CosineSimilarity(dim=1)
    num_img = x.size(0)
    # cos_sim = cos(x.view(-1), y.view(-1))
    cos_sim = cos(x.view(num_img, -1), y.view(num_img, -1))
    # if metric == 'mean'
    return torch.mean(1/ math.pi * torch.acos(cos_sim))

def TripletLoss(anchor, positive, negative, margin=0.2):
    cos = nn.CosineSimilarity(dim=1)
    num_img = anchor.size(0)
    d_ap = 1 - cos(anchor.view(num_img, -1), positive.view(num_img, -1))
    d_an = 1 - cos(anchor.view(num_img, -1), negative.view(num_img, -1))
    dist = d_ap - d_an + margin
    return torch.mean(torch.max(dist, torch.zeros_like(dist)))