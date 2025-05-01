import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset_loader import Middlebury_dataset
from sgtnet import SGTNet
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SwinIR model on depth data')
    parser.add_argument('--scale', type=int, default=16, help='Upscaling factor')
    parser.add_argument('--img_size', type=int, default=256, help='patch size')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--pretrained_path', type=str, default='./checkpoints/x16_256.pth', help='Path to pretrained model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/Lu', help='Path to dataset directory')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = SGTNet(
        img_size=args.img_size,
        upscale=args.scale,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2
    ).to(device)
    net.load_state_dict(torch.load(args.pretrained_path, map_location=device))
    net.to(device)
    net.eval()

    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset_name = os.path.basename(args.dataset_dir)

    if dataset_name in ['Middlebury', 'Lu']:
        dataset = Middlebury_dataset(root_dir=args.dataset_dir, scale=args.scale, transform=data_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    save_dir = os.path.join(args.results_dir, 'saved_depth_maps')
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            guidance = data['guidance'].to(device)
            lr = data['lr'].to(device)
            out = net(guidance, lr)

            lr_depth = lr[0, 0].cpu().numpy()
            pred_depth = out[0, 0].cpu().numpy()

            plt.imsave(f"{save_dir}/lr_depth_{idx}.png", lr_depth, cmap='jet')
            plt.imsave(f"{save_dir}/pred_depth_{idx}.png", pred_depth, cmap='jet')

if __name__ == "__main__":
    main()
