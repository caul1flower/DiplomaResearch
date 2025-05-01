import os
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import NYU_v2_datset
from sgtnet import SGTNet, load_pretrained_swin2sr
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import log_model_summary, visualize_depth_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SwinIR model on depth data')
    parser.add_argument('--scale', type=int, default=16, help='Upscaling factor')
    parser.add_argument('--img_size', type=int, default=256, help='patch size')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--pretrained_path', type=str, default='./checkpoints/x16_256.pth', help='Path to pretrained model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--use_Swin2SR_pretrained', type=bool, default=False, help='Use Swin2SR pretrained model for depth and RGB')
    parser.add_argument('--swini2sr_pretrained_path', type=str, default='./checkpoints/Swin2SR_Lightweight.pth', help='Use Swin2SR pretrained model for depth and RGB')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/nyu_data', help='Path to dataset directory')
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
        'epochs': args.epochs,
        'lr': args.lr,
        'dataset_dir': args.dataset_dir,
        'results_dir': args.results_dir,
        'pretrained_path': args.pretrained_path
    }
    data_transforms = transforms.Compose([transforms.ToTensor()])
    experiment_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"s{config['scale']}_{config['img_size']}_"
        f"lr{config['lr']}_bs{config['batch_size']}_"
        f"epochs{config['epochs']}_swinir_depth"
    )
    results_path = os.path.join(config['results_dir'], experiment_name)
    os.makedirs(results_path, exist_ok=True)
    writer = SummaryWriter(log_dir=results_path)
    
    train_dataset_nyu = NYU_v2_datset(
        root_dir=config['dataset_dir'],
        scale=config['scale'],
        patch_size=config['img_size'],
        train=True,
        transform=data_transforms
    )
    test_dataset_nyu = NYU_v2_datset(
        root_dir=config['dataset_dir'],
        scale=config['scale'],
        patch_size=config['img_size'],
        train=False,
        transform=transforms.ToTensor()
    )
    train_loader = DataLoader(
        train_dataset_nyu,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        test_dataset_nyu,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )

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

    if args.use_Swin2SR_pretrained:
        net = load_pretrained_swin2sr(args.pretrained_path, net, encoder_type='rgb')
        net = load_pretrained_swin2sr(args.pretrained_path, net, encoder_type='deth')

    model_name = log_model_summary(writer, model, config)
    writer.add_text('Pretrained Model', f'Pretrained model loaded from: {config["pretrained_path"]}')
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_rmse = float('inf')
    train_losses = []
    val_metrics = {'loss': [], 'rmse': []}
    fixed_sample = None
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")

        for batch in progress_bar:
            guidance = batch['guidance'].to(device)
            lr_depth = batch['lr'].to(device)
            gt_depth = batch['gt'].to(device)
            optimizer.zero_grad()
            pred_depth = model(guidance, lr_depth)
            loss = criterion(pred_depth, gt_depth)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * guidance.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = epoch_loss / len(train_dataset_nyu)
        train_losses.append(avg_train_loss)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        model.eval()
        val_loss = 0.0
        rmse = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch in val_bar:
                guidance = batch['guidance'].to(device)
                lr_depth = batch['lr'].to(device)
                gt_depth = batch['gt'].to(device)

                pred_depth = model(guidance, lr_depth)
                loss = criterion(pred_depth, gt_depth)
                val_loss += loss.item()

                mse = torch.mean((pred_depth - gt_depth)**2)
                batch_rmse = torch.sqrt(mse)
                rmse += batch_rmse.item()

                val_bar.set_postfix({'val_loss': loss.item(), 'rmse': batch_rmse.item()})

        avg_val_loss = val_loss / len(val_loader)
        avg_rmse = rmse / len(val_loader)
        val_metrics['loss'].append(avg_val_loss)
        val_metrics['rmse'].append(avg_rmse)

        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('RMSE', avg_rmse, epoch)

        scheduler.step(avg_val_loss)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            torch.save(model.state_dict(), os.path.join(results_path, 'best_model.pth'))

        if fixed_sample is None:
            fixed_sample = next(iter(val_loader))

        with torch.no_grad():
            fixed_guidance = fixed_sample['guidance'][0].unsqueeze(0).to(device)
            fixed_lr_depth = fixed_sample['lr'][0].unsqueeze(0).to(device)
            fixed_gt_depth = fixed_sample['gt'][0].unsqueeze(0).to(device)

            fixed_pred_depth = model(fixed_guidance, fixed_lr_depth)

            visualize_depth_predictions(
                fixed_guidance,
                fixed_lr_depth,
                fixed_pred_depth,
                fixed_gt_depth,
                epoch,
                results_path,
                writer,
                model_name
            )
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | RMSE: {avg_rmse:.4f}")

    torch.save(model.state_dict(), os.path.join(results_path, 'final_model.pth'))
    writer.close()

    # Plotting training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_metrics['loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_metrics['rmse'], label='RMSE')
    plt.title('Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'rmse_curve.png'))
    plt.close()

    print("Training completed!")

if __name__ == "__main__":
    main()
