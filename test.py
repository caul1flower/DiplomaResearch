import os
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sgtnet import SGTNet
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_loader import NYU_v2_datset, Middlebury_dataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SwinIR model on depth data')
    parser.add_argument('--scale', type=int, default=8, help='Upscaling factor')
    parser.add_argument('--img_size', type=int, default=256, help='patch size')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--pretrained_path', type=str, default='./checkpoints/x8_256.pth', help='Path to pretrained model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/Lu', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration dictionary
    config = {
        'scale': args.scale,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'dataset_dir': args.dataset_dir,
        'results_dir': args.results_dir,
        'model_path': args.pretrained_path,
    }

    os.makedirs(config['results_dir'], exist_ok=True)

    model = SGTNet(
        img_size=config['img_size'],
        upscale=config['scale'],
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2
    ).to(device)


    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.to(device)
    model.eval()


    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset_name = config['dataset_dir'].split('/')[-1]

    if dataset_name == 'nyu_data':
        dataset = NYU_v2_datset(
            root_dir=config['dataset_dir'], 
            scale=config['scale'], 
            transform=data_transform, 
            train=False
        )
        test_minmax = np.load(f"{config['dataset_dir']}/test_minmax.npy")
    elif dataset_name in ['Middlebury', 'Lu']:
        dataset = Middlebury_dataset(
            root_dir=config['dataset_dir'], 
            scale=config['scale'], 
            transform=data_transform
        )

    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0
    )

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    results = []

    model_name = os.path.basename(config['model_path']).split('.')[0]

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing")
        for idx, data in enumerate(progress_bar):
            guidance = data['guidance'].to(device)
            lr = data['lr'].to(device)
            gt = data['gt'].to(device)

            out = model(guidance, lr)

            if dataset_name == 'nyu_data':
                minmax = test_minmax[:, idx]
                minmax = torch.from_numpy(minmax).to(device)
                gt_normalized = (gt[0, 0] - minmax[1]) / (minmax[0] - minmax[1])
                out_normalized = (out[0, 0] - minmax[1]) / (minmax[0] - minmax[1])
            elif dataset_name in ['Middlebury', 'Lu']:  # Middlebury or Lu
                gt_normalized = (gt[0, 0] - gt[0, 0].min()) / (gt[0, 0].max() - gt[0, 0].min())
                out_normalized = (out[0, 0] - out[0, 0].min()) / (out[0, 0].max() - out[0, 0].min())

            psnr = psnr_metric(gt_normalized.unsqueeze(0).unsqueeze(0), 
                              out_normalized.unsqueeze(0).unsqueeze(0)).item()
            ssim = ssim_metric(gt_normalized.unsqueeze(0).unsqueeze(0), 
                              out_normalized.unsqueeze(0).unsqueeze(0)).item()

            results.append([idx, psnr, ssim])
            progress_bar.set_postfix({'PSNR': psnr, 'SSIM': ssim})

    avg_psnr = np.mean([row[1] for row in results])
    avg_ssim = np.mean([row[2] for row in results])

    csv_filename = f"{config['results_dir']}/metrics_results_{dataset_name}_{model_name}.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Image Index", "PSNR", "SSIM"])
        csv_writer.writerows(results)
        csv_writer.writerow(["Average", 
                            round(avg_psnr, 3), 
                            round(avg_ssim, 3)])
        csv_writer.writerow(["Dataset", dataset_name, 
                            "Scale", config['scale'], 
                            "Img_size", config['img_size'], 
                            "Model", model_name])
    
    print(f"Results saved to {csv_filename}")
    print(f"Average Metrics - PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.3f}")

if __name__ == "__main__":
    main()
