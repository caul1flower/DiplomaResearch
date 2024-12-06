import torch
import torch.nn.functional as F
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from models.SGNet import SGNet
from data.nyu_dataloader import NYU_v2_datset


def calc_rmse(a, b, minmax):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    a = a*(minmax[0]-minmax[1]) + minmax[1]
    b = b*(minmax[0]-minmax[1]) + minmax[1]
    a = a * 100
    b = b * 100
    return torch.sqrt(torch.mean(torch.pow(a-b,2)))

def calc_psnr(a, b, minmax, max_value=255.0):
    rmse = calc_rmse(a, b, minmax)
    if rmse.item() == 0:
        return 100.0
    max_value = (minmax[0] - minmax[1]) * 100
    return 20 * torch.log10( max_value / rmse)

def create_window(window_size, channel):
    kernelX = cv2.getGaussianKernel(window_size, 1.5)
    window = kernelX * kernelX.T
    window = torch.from_numpy(window).float()
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size)
    return window

def calc_ssim(a, b, minmax, window_size = 11):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    a = a*(minmax[0]-minmax[1]) + minmax[1]
    b = b*(minmax[0]-minmax[1]) + minmax[1]
    if a.ndim == 2:
        a = a.unsqueeze(0).unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0).unsqueeze(0)

    padding = window_size // 2
    channel = a.size(1)

    window = create_window(window_size, channel).to(a.device)

    mu_a = F.conv2d(a, window, padding=padding, groups=channel)
    mu_b = F.conv2d(b, window, padding=padding, groups=channel)

    sigma_a = F.conv2d(a * a, window, padding=padding, groups=channel) - mu_a**2
    sigma_b = F.conv2d(b * b, window, padding=padding, groups=channel) - mu_b**2
    sigma_ab = F.conv2d(a * b, window, padding=padding, groups=channel) - mu_a * mu_b

    L = (minmax[0] - minmax[1])
    C1 = (L * 0.01)**2
    C2 = (L * 0.03)**2
    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / (
        (mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2)
    )
    return ssim_map.mean().item()


parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=16, help='scale factor')
parser.add_argument("--num_feats", type=int, default=40, help="channel number of the middle hidden layer")
parser.add_argument("--root_dir", type=str, default='/datapath/nyu_data', help="root dir of dataset")
parser.add_argument("--model_dir", type=str, default="/SGNet/ckpt/SGNet_X16_R.pth", help="path of model")
parser.add_argument("--results_dir", type=str, default='/SGNet/results', help="root dir of results")
opt = parser.parse_args()

net = SGNet(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale)
net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

data_transform = transforms.Compose([transforms.ToTensor()])
dataset_name = opt.root_dir.split('/')[-1]

if dataset_name == 'nyu_data':
    dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
    test_minmax = np.load(f'{opt.root_dir}/test_minmax.npy')
    rmse = np.zeros(len(dataset))
    psnr_list = []
    ssim_list = []

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

with torch.no_grad():
    net.eval()
    if dataset_name == 'nyu_data':
        for idx, data in enumerate(dataloader):
            guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
            out, _ = net((guidance, lr))
            minmax = test_minmax[:, idx]
            minmax = torch.from_numpy(minmax).to(device)
            rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)
            psnr = calc_psnr(gt[0, 0], out[0, 0], minmax)
            ssim = calc_ssim(gt[0, 0], out[0, 0], minmax, 11)

            psnr_list.append(psnr.item())
            ssim_list.append(ssim)
            print(f"Image {idx}: RMSE={rmse[idx]:.4f}, PSNR={psnr:.4f}, SSIM={ssim:.4f}")

        print(f"Average RMSE: {rmse.mean():.4f}")
        print(f"Average PSNR: {np.mean(psnr_list):.4f}")
        print(f"Average SSIM: {np.mean(ssim_list):.4f}")
