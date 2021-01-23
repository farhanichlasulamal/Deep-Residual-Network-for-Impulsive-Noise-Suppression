import torch
import torch.nn as nn


class Conv2D(nn.Module):

    def __init__(self, in_feats, out_feats, kernel_size):
        super(Conv2D, self).__init__()
        pad = kernel_size//2
        self.conv = nn.Conv2d(in_feats, out_feats, kernel_size, padding=pad)
        self.act = nn.ReLU()


    def forward(self, x):
        y = self.conv(x)
        y = self.act(y)
        return y

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  RESBLOCK Pre-Activation
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, num_feats,kernel_size):
        super(ResBlock, self).__init__()
        layers = [
          nn.BatchNorm2d(num_feats),
          nn.Conv2d(num_feats, num_feats,kernel_size,padding=kernel_size//2),
          nn.ReLU(),
          nn.BatchNorm2d(num_feats),
          nn.Conv2d(num_feats, num_feats,kernel_size,padding=kernel_size//2),
          nn.ReLU()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x) + x
        return y


class DenoiserNet(nn.Module):

    def __init__(self, iscolor, noise_lv, num_feats, depth, kernel_size, comment):
        super(DenoiserNet, self).__init__()
        (in_feats, mode) = (3, 'color') if iscolor else (1, 'gray')
        self.name = mode + '_noise_lv' +str(noise_lv)+'_numfeat'+ str(num_feats) +'_depth' +str(depth) +'_kernel' +str(kernel_size)+'_'+comment
        #Silakan di edit
        self.conv_in = Conv2D(in_feats, num_feats, kernel_size)
        conv_hid = [ResBlock(num_feats, kernel_size) for i in range(depth)]
        self.conv_hid = nn.Sequential(*conv_hid)
        self.conv_out = nn.Conv2d(num_feats, in_feats, kernel_size, padding=kernel_size//2)


    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_hid(y)
        y = self.conv_out(y)
        return y

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  RESBLOCK ORIGINAL
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class UnitResNet(nn.Module):
    def __init__(self, num_feats,kernel_size):
        super(UnitResNet, self).__init__()
        layers = [
          nn.Conv2d(num_feats, num_feats,kernel_size,padding=kernel_size//2),
          nn.BatchNorm2d(num_feats),
          nn.ReLU(),
          nn.Conv2d(num_feats, num_feats,kernel_size,padding=kernel_size//2),
          nn.BatchNorm2d(num_feats)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x) + x
        return y

class ResNet(nn.Module):
    def __init__(self, num_feats, kernel_size):
        super(ResNet, self).__init__()
        activation_layers = [
          UnitResNet(num_feats, kernel_size),
          nn.ReLU()
        ]
        self.activation_layers = nn.Sequential(*activation_layers)

    def forward(self, x):
        y = self.activation_layers(x)
        return y

class DenoiserNetORG(nn.Module):

    def __init__(self, iscolor, noise_lv, num_feats, depth, kernel_size, comment):
        super(DenoiserNetORG, self).__init__()
        (in_feats, mode) = (3, 'color') if iscolor else (1, 'gray')
        self.name = mode + '_noise_lv' +str(noise_lv)+'_numfeat'+ str(num_feats) +'_depth' +str(depth) +'_kernel' +str(kernel_size)+'_'+comment
        #Silakan di edit
        self.conv_in = Conv2D(in_feats, num_feats, kernel_size)
        conv_hid = [ResNet(num_feats, kernel_size) for i in range(depth)]
        self.conv_hid = nn.Sequential(*conv_hid)
        self.conv_out = nn.Conv2d(num_feats, in_feats, kernel_size, padding=kernel_size//2)


    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_hid(y)
        y = self.conv_out(y)
        return y