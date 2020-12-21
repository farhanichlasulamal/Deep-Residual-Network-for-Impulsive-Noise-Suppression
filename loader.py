import os
import torch
import numpy as np
from sporco import util

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor

def polute_img(pil_img, noise_lv):
    np_img = np.array(pil_img)
    np_img = util.spnoise(np_img, noise_lv, 0, 255)
    return Image.fromarray(np_img)

class ImageDataset(Dataset):

    def __init__(self, sets='train', iscolor=False, noise_lv=0.1):
        super(ImageDataset, self).__init__()
        mode = 'color' if iscolor else 'gray'
        self.noise_lv = noise_lv
        self.clean_patches = list()

        set_path = os.path.join('datasets', sets +' ' +mode)

        clean_path = os.path.join(set_path, 'clean')
        
        imgs_dir = os.listdir(clean_path)
        for folder in imgs_dir:
            clean_folder = os.path.join(clean_path, folder)
        
            patch_dir = os.listdir(clean_folder)
            for patch in patch_dir:
                clean_patch = os.path.join(clean_folder, patch)
                self.clean_patches.append(clean_patch)
                

    def __getitem__(self, index):
        clean = self.clean_patches[index]
        clean = Image.open(clean)
        noisy = polute_img(clean, self.noise_lv)
        clean_img = ToTensor()(clean)
        noisy_img = ToTensor()(noisy)
        return clean_img, noisy_img


    def __len__(self):
        return len(self.clean_patches)
