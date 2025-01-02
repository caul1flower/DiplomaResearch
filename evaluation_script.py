import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models.SGNet import SGNet
from data.nyu_dataloader import NYU_v2_datset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import csv
import os

def calc_rmse(a, b, minmax):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    a = a*(minmax[0]-minmax[1]) + minmax[1]
    b = b*(minmax[0]-minmax[1]) + minmax[1]
    a = a * 100
    b = b * 100
    return torch.sqrt(torch.mean(torch.pow(a-b,2)))

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=16, help='scale factor')
parser.add_argument("--num_feats", type=int, default=40, help="channel number of the middle hidden layer")
parser.add_argument("--root_dir", type=str, default='./datapath/nyu_data', help="root dir of dataset")
parser.add_argument("--model_dir", type=str, default="./SGNet/ckpt/SGNet_X16_R.pth", help="path of model")
parser.add_argument("--results_dir", type=str, default='./results', help="root dir of results")
opt = parser.parse_args()

net = SGNet(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale)
net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

data_transform = transforms.Compose([transforms.ToTensor()])
dataset_name = opt.root_dir.split('/')[-1]
model_name = opt.model_dir.split('/')[-1].split('.')[0]

if dataset_name == 'nyu_data':
    dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
    test_minmax = np.load(f'{opt.root_dir}/test_minmax.npy')
    results = []

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

psnr_metric = PeakSignalNoiseRatio().to(device)
ssim_metric = StructuralSimilarityIndexMeasure().to(device)


with torch.no_grad():
    net.eval()
    if dataset_name == 'nyu_data':
        for idx, data in enumerate(dataloader):
            guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
            out, _ = net((guidance, lr))
            minmax = test_minmax[:, idx]
            minmax = torch.from_numpy(minmax).to(device)
            rmse = calc_rmse(gt[0, 0], out[0, 0], minmax)
            
            gt_normalized = (gt[0, 0] - minmax[1]) / (minmax[0] - minmax[1])
            out_normalized = (out[0, 0] - minmax[1]) / (minmax[0] - minmax[1])

            psnr = psnr_metric(gt_normalized.unsqueeze(0).unsqueeze(0), out_normalized.unsqueeze(0).unsqueeze(0)).item()
            ssim = ssim_metric(gt_normalized.unsqueeze(0).unsqueeze(0), out_normalized.unsqueeze(0).unsqueeze(0)).item()

            results.append([idx, rmse, psnr, ssim])

        os.makedirs(opt.results_dir, exist_ok=True)
        with open(f"{opt.results_dir}/metrics_results_{dataset_name}_{model_name}.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Image Index", "RMSE", "PSNR", "SSIM"])
            csv_writer.writerows(results)
            csv_writer.writerow(["Average", round(np.mean([row[1] for row in results]), 3), 
                                            round(np.mean([row[2] for row in results]), 3), 
                                            round(np.mean([row[3] for row in results]), 3)])
            csv_writer.writerow(["Dataset", dataset_name, "Scale", opt.scale, "Num_Feats", opt.num_feats, "Model", model_name])

        print(f"Results saved to {opt.results_dir}/metrics_results_{dataset_name}_{model_name}.csv")
