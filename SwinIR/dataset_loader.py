import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset

class NYU_v2_dataset(Dataset):
    """NYUDataset with configurable image size."""

    def __init__(self, root_dir, scale=8, train=True, transform=None, img_size=128):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            img_size (int): Size of the output images (assumes square images)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train
        self.img_size = img_size
        
        if train:
            self.depths = np.load(f'{root_dir}/train_depth_split.npy')
            self.images = np.load(f'{root_dir}/train_images_split.npy')
        else:
            self.depths = np.load(f'{root_dir}/test_depth.npy')
            self.images = np.load(f'{root_dir}/test_images_v2.npy')

    def __len__(self):
        return self.depths.shape[0]

    def resize_image(self, image, size):
        """Safely resize an image array."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        image = np.squeeze(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((size, size), Image.BICUBIC)

        return np.array(pil_img)

    def resize_depth(self, depth, size):
        """Safely resize a depth array."""

        depth = np.squeeze(depth)
        depth = depth.astype(np.float32)

        pil_depth = Image.fromarray(depth)
        pil_depth = pil_depth.resize((size, size), Image.BICUBIC)

        depth = np.array(pil_depth)
        return np.expand_dims(depth, axis=-1)

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        
        if self.train:
            image, depth = get_patch(
                img=image, 
                gt=np.expand_dims(depth, 2), 
                patch_size=self.img_size
            )
            image, depth = augment(img=image, gt=depth)
        else:
            image = self.resize_image(image, self.img_size)
            depth = self.resize_depth(depth, self.img_size)

        s = self.scale
        lr_depth = self.resize_depth(depth, self.img_size // s)

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(depth).float()
            lr = self.transform(lr_depth).float()

        sample = {'guidance': image, 'lr': lr, 'gt': depth}
        
        return sample

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
