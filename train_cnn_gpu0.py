import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from tqdm import tqdm
from PIL import Image
from math import ceil

from utils import *
from models.denoisernet import DenoiserNet, DenoiserNetORG
from models.criterion import TotalVariation
from loader import ImageDataset

SEED = 3478
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

EPOCH = 3
NUM_BATCH = 0
ITERATION = None#NUM_BATCH * (0 if EPOCH == None else EPOCH) + 1
NUM_EPOCHS = 25
DEPTH = 8
NUM_FEATS = 16
KERNEL_SIZE = 3
IS_COLOR = True
NOISE_LV = 0.5
COMMENT = 'no tv - Resblock PRE-ACTIVATION' #Kasih judul percobaan disini
GPU_ID = 0


def train(net, loss_fn, optimizer, train_loader, val_loader, epoch, log_dir):
    global ITERATION
    ITERATION = NUM_BATCH * (epoch -1) + 1
    net.train()
    train_bar = tqdm(train_loader)
    batch_sizes = 0
    total_loss  = 0

    for clean, dither in train_bar:
        batch_size = clean.size(0)
        batch_sizes += batch_size

        clean = clean.cuda(GPU_ID)
        dither = dither.cuda(GPU_ID)

        net.zero_grad()
        img_rec = net(dither)
        
        loss = loss_fn(img_rec, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        train_bar.set_description(desc='itr:%d [%3d/%3d] Loss: %.8f' %(
                ITERATION,
                epoch, NUM_EPOCHS,
                total_loss / batch_sizes
            )
        )
        write_csv(log_dir+'/loss_log_{}.csv'.format(epoch), [ITERATION, loss.item()])
        ITERATION = ITERATION + 1
    torch.cuda.empty_cache()


def main(checkpoint=None, batch_size=32, lr=0.001):

    train_set = ImageDataset('train', IS_COLOR, NOISE_LV)
    val_set = ImageDataset('valid', IS_COLOR, NOISE_LV)
    test_set = ImageDataset('test', IS_COLOR, NOISE_LV)

    train_loader = DataLoader(dataset=train_set, num_workers=16, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=16, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_set, num_workers=16, batch_size=1, shuffle=False)
    global NUM_BATCH
    NUM_BATCH = ceil(len(train_loader.dataset) /batch_size)

    net = DenoiserNet(IS_COLOR, NOISE_LV, NUM_FEATS, DEPTH, KERNEL_SIZE, COMMENT)
    net_name = net.name
    #net.apply(weights_init) 

    mse_loss = nn.MSELoss().cuda(GPU_ID)
    # tv_loss = TotalVariation().cuda(GPU_ID)

    # def loss_fn(rec, gt):
    #     loss = mse_loss(rec, gt) + 2E-4 * tv_loss(rec)
    #     return loss

    net =  net.cuda(GPU_ID)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    log_dir = 'logs/'+net_name
    checkpoint_dir = 'checkpoint/'+net_name

    test_dir = 'test imgs/'+net_name
    val_dir = 'val imgs/'+net_name
    
    mkdir(log_dir)
    mkdir(checkpoint_dir)
    mkdir(test_dir)
    mkdir(val_dir)
    
    
    if checkpoint is not None:
        load_checkpoint(checkpoint_dir, net, optimizer, EPOCH)
        lr = get_lr(optimizer)
    else:
        checkpoint = 0

    for epoch in range(checkpoint+1, NUM_EPOCHS+1):
        if epoch % 3 == 0:
            lr = lr / 2.
            change_lr(optimizer, lr)
        train(net, mse_loss, optimizer, train_loader, val_loader, epoch, log_dir)
        save_checkpoint(checkpoint_dir, net, optimizer, epoch)
        with torch.no_grad():
            if epoch % 5 == 0 or epoch < 5:
                psnr_val = demo(net, val_dir, epoch, val_loader, True)
                psnr_test = demo(net, test_dir, epoch, test_loader, True)
                write_csv(log_dir + '/psnr_test_log.csv', [ITERATION, psnr_test])
            else:
                psnr_val = demo(net, val_dir, epoch, val_loader, False)
            write_csv(log_dir + '/psnr_val_log.csv', [ITERATION, psnr_val])


def test(checkpoint=None):

    test_set = ImageDataset('test', IS_COLOR, NOISE_LV)

    test_loader = DataLoader(dataset=test_set, num_workers=16, batch_size=1, shuffle=False)
    global NUM_BATCH
    NUM_BATCH = len(test_loader.dataset)

    net = DenoiserNet(IS_COLOR, NOISE_LV, NUM_FEATS, DEPTH, KERNEL_SIZE, COMMENT)
    net_name = net.name
    
    net =  net.cuda(GPU_ID)
    checkpoint_dir = 'checkpoint/'+net_name
    test_dir = 'test imgs/'+net_name
    mkdir(test_dir)
      
    load_checkpoint(checkpoint_dir, net, None, checkpoint)
    with torch.no_grad():
        demo(net, test_dir, checkpoint, test_loader, save=False)
    pass


def save_feats(net, path_dir, epoch, loader, save=False):
    net.eval()
    psnr_sum = 0
    j = 1
    for clean, dither in loader:
        embeded_pad, h, w = check_size(dither, 8)
        rec_t = net(embeded_pad.cuda(GPU_ID))
        # Remove pad
        rec_t = rec_t[:,:,:h,:w]
        chan = rec_t.shape[1]
        i = 0
        for c in range(chan):
            rec = rec_t[:,c:c+1,:,:]
            rec = feat2img(rec)
            i += 1
            rec.save('{}/{}_{}_{}.png'.format(path_dir, j, ITERATION, i))
        torch.cuda.empty_cache()
        j = j + 1
    pass

def demo(net, path_dir, epoch, loader, save=False):
    net.eval()
    psnr_sum = 0
    i = 0
    for clean, noisy in loader:
        #noisy, h, w = noisy
        rec = net(noisy.cuda(GPU_ID))
        # Remove pad
        #rec = rec[:,:,:h,:w]
        rec = tensor2img(rec)
        clean = tensor2img(clean)
        mse = ((np.array(clean, dtype='float') - np.array(rec, dtype='float'))**2).mean()
        psnr = np.log10(255**2/mse) * 10
        psnr_sum += psnr
        i += 1
        if save:
            rec.save('{}/{}_{}_{}.png'.format(path_dir, i, ITERATION, psnr))
        torch.cuda.empty_cache()
    print('PNSR: {} {}'.format(psnr_sum/len(loader.dataset), len(loader.dataset)))
    return psnr_sum/len(loader.dataset)



if __name__ == '__main__':
    main()
    # test(24)
    pass