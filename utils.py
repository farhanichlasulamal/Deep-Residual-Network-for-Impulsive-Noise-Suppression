import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import utils

def visTensor(tensor, ch=0, allkernels=False, nrow=16, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding, pad_value=255)
    plt.tight_layout()
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

def mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def save_checkpoint(checkpoint_dir, model, optim, epoch):
    model_out_path = checkpoint_dir + "/epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model, "optim":optim}
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def load_checkpoint(checkpoint_dir, net, optimizer, epoch):
    model_folder = checkpoint_dir + "/epoch_{}.pth".format(epoch)
    weights = torch.load(model_folder)
    net.load_state_dict(weights['model'].state_dict())
    if not(optimizer is None):
        optimizer.load_state_dict(weights['optim'].state_dict())


def weights_init(m, std=0.001):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, std)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=0.001): #0.001
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)


def change_lr(optim, lr):
    for param in optim.param_groups:
        param['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def write_csv(file, data):
    with open(file, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


def tensor2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor =  np.squeeze(tensor)
    if len(tensor.shape) == 3:
        tensor = np.moveaxis(tensor, 0, 2)
    tensor = tensor * 255
    tensor = tensor.clip(0, 255).astype(np.uint8)
    
    img = Image.fromarray(tensor)
    return img


def feat2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor =  np.squeeze(tensor)
    tensor = np.interp(tensor, (tensor.min(), tensor.max()), (0, 255))
    #tensor = np.moveaxis(tensor, 0, 2)
    tensor = tensor.astype(np.uint8)
    img = Image.fromarray(tensor)
    return img


def check_size(tensor, p=8):
    b, c, h, w =  tensor.shape
    h_pad =  (h % p)
    w_pad =  (w % p)
    tensor = F.pad(tensor, [0, w_pad, 0, h_pad],'reflect')
    return tensor, h, w