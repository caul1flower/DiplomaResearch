import os
from matplotlib import pyplot as plt


def log_model_summary(writer, model, config):
    """Log model architecture and configuration to TensorBoard"""
    model_name = type(model).__name__
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Log model architecture
    writer.add_text('Model/Name', model_name)
    writer.add_text('Model/TotalParameters', f'{total_params:,}')
    writer.add_text('Model/TrainableParameters', f'{trainable_params:,}')
    
    # Log training configuration
    for key, value in config.items():
        writer.add_text(f'Config/{key}', str(value))
    
    return model_name


def visualize_depth_predictions(guidance, lr_depth, pred_depth, gt_depth, epoch, save_dir, writer, model_name):
    """Visualize and save depth prediction results"""
    # Convert tensors to numpy for visualization
    guidance_np = guidance[0, 0].cpu().numpy()  # Take first image, first channel
    lr_depth_np = lr_depth[0, 0].cpu().numpy()
    pred_depth_np = pred_depth[0, 0].cpu().numpy()
    gt_depth_np = gt_depth[0, 0].cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot images
    axes[0].imshow(guidance_np, cmap='gray')
    axes[0].set_title('Guidance (RGB)')
    axes[0].axis('off')
    
    axes[1].imshow(lr_depth_np, cmap='jet')
    axes[1].set_title('LR Depth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_depth_np, cmap='jet')
    axes[2].set_title('Predicted Depth')
    axes[2].axis('off')
    
    axes[3].imshow(gt_depth_np, cmap='jet')
    axes[3].set_title('GT Depth')
    axes[3].axis('off')
    
    # Save figure
    save_path = os.path.join(save_dir, f'depth_vis_epoch_{epoch}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Log to TensorBoard
    writer.add_figure(f'Depth Visualization/{model_name}', fig, epoch)
