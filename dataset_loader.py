import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import os

class NYU_v2_datset(Dataset):
    """NYUDataset."""

    def __init__(self, root_dir, scale=8, train=True, transform=None, patch_size=128, window_size=8):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.

        """
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train
        self.patch_size = patch_size
        self.window_size = window_size

        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            self.images = np.load('%s/train_images_split.npy'%root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            self.images = np.load('%s/test_images_v2.npy'%root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        if self.train:
            image, depth = get_patch(img=image, gt=np.expand_dims(depth,2), patch_size=self.patch_size)
            image, depth = augment(img=image, gt=depth)

        else:
            image = modcrop(image, self.scale*self.window_size)
            depth = modcrop(depth, self.scale*self.window_size)

        h, w = depth.shape[:2]
        s = self.scale
        lr = np.array(Image.fromarray(depth.squeeze()).resize((w//s,h//s), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(depth).float()
            lr = self.transform(np.expand_dims(lr,2)).float()

        sample = {'guidance': image, 'lr': lr, 'gt': depth}
        return sample


class Middlebury_dataset(Dataset):
    """Middlebury Dataset."""

    def __init__(self, root_dir, scale=8, transform=None, window_size=8):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.transform = transform
        self.scale = scale
        self.window_size = window_size

        self.GTs = []
        self.RGBs = []
        
        list_dir = os.listdir(root_dir)
        for name in list_dir:
            if name.find('output_color') > -1:
                self.RGBs.append('%s/%s' % (root_dir, name))
            elif name.find('output_depth') > -1:
                self.GTs.append('%s/%s' % (root_dir, name))
        self.RGBs.sort()
        self.GTs.sort()

    def __len__(self):
        return len(self.GTs)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.RGBs[idx]))
        gt = np.array(Image.open(self.GTs[idx]))
        assert gt.shape[0] == image.shape[0] and gt.shape[1] == image.shape[1]
        scale = self.scale  
        image = modcrop(image, scale*self.window_size)
        gt = modcrop(gt, scale*self.window_size)

        h, w = gt.shape[0], gt.shape[1]

        lr = np.array(Image.fromarray(gt).resize((w//scale,h//scale),Image.BICUBIC)).astype(np.float32)
        gt = gt / 255.0
        image = image / 255.0
        lr = lr / 255.0
        

        if self.transform:
            image = self.transform(image).float()
            gt = self.transform(np.expand_dims(gt,2)).float()
            lr = self.transform(np.expand_dims(lr,2)).float()

        sample = {'guidance': image, 'lr': lr, 'gt': gt}
        return sample

def calc_rmse(a, b, minmax):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    a = a*(minmax[0]-minmax[1]) + minmax[1]
    b = b*(minmax[0]-minmax[1]) + minmax[1]
    a = a * 100
    b = b * 100
    return torch.sqrt(torch.mean(torch.pow(a-b,2)))

def modcrop(image, modulo):
    """Crop image to make dimensions divisible by modulo."""
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo
    return image[:h, :w]


def augment(img, gt, hflip=True, rot=True):
    """Augment the image and ground truth with flips."""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    
    if hflip:
        img = img[:, ::-1, :].copy()
        gt = gt[:, ::-1, :].copy()
    if vflip:
        img = img[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()
    
    return img, gt

def get_patch(img, gt, patch_size=128):
    """Extract a random patch from the image and ground truth."""
    th, tw = img.shape[:2]
    tp = round(patch_size)
    
    tp = min(tp, th, tw)
    
    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))
    
    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]
