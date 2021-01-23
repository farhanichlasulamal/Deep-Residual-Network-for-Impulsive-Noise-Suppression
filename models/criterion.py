import torch
from torch import nn

class TotalVariation(nn.Module):
    def __init__(self, reduce='mean'):
        super(TotalVariation, self).__init__()
        assert reduce == 'sum' or reduce == 'mean', 'Unsupported reduce [{}]'.format(reduce)
        self.reduce = reduce

    def forward(self, imgs):
        abs_difh = torch.abs(imgs[:,:,1:,:]-imgs[:,:,:-1,:])
        abs_difw = torch.abs(imgs[:,:,:,1:]-imgs[:,:,:,:-1])

        if self.reduce == 'sum':
            loss = abs_difh.sum() + abs_difw.sum()
            return loss
        else:
            batch, channels, height, width = imgs.size()
            count_h = channels * (height-1) * width
            count_w = channels * height * (width-1)
            loss = abs_difh.sum() / count_h + abs_difw.sum() / count_w
            loss = loss / batch
            return loss