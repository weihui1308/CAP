import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn as nn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from unet_model import UNet
from dataloader import BasicDataset
from ispNet_val import evaluate
from PIL import Image
from matplotlib import pyplot as plt

import pytorch_ssim


def train_model(
        model,
        device,
        epochs: int = 3000,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = True,
        val_percent: float = 0.1,
        ):
    
    
    # load data
    root_dir = "image-pair-file-path"
    param_file = 'parameter-file-path'
    
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
    ])
    
    images_patch = os.listdir(root_dir)
    train_images, val_images = train_test_split(images_patch, test_size=val_percent, random_state=7)
    
    train_dataset = BasicDataset(root_dir, train_images, param_file, transform=transform)
    val_dataset = BasicDataset(root_dir, val_images, param_file, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    n_train = len(train_dataloader)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterionSSIM = pytorch_ssim.SSIM()
    criterionMSE = nn.MSELoss()
    
    maxx = 1000
    train_loss_list = []
    val_loss_list = []
    dir_checkpoint = Path('/home/xxx/checkpoints/')
    os.makedirs('/home/xxx/checkpoints/', exist_ok=True)
    # training
    for epoch in range(0, epochs):
        epoch_loss = 0
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar, total=n_train, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, batch in pbar:  # batch -------------------------------------------------------------
            optimizer.zero_grad()
            
            orig_image, target_img, param = batch['orig_image'], batch['target_img'], batch['param']
            orig_image = orig_image.to(device=device, dtype=torch.float32)
            target_img = target_img.to(device=device, dtype=torch.float32)
            param = param.to(device=device, dtype=torch.float32)

            pred = model(orig_image, param)
            
            loss = (1 - criterionSSIM(pred, target_img)) * 0.5 + criterionMSE(pred, target_img) * 0.5

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.detach().cpu().item()

        train_loss = epoch_loss / n_train
        val_loss = evaluate(model, val_dataloader, device)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        
        if epoch % 10 == 0:
            img_save = pred[0].detach()
            im = transforms.ToPILImage()(img_save)
            best_pred_dir = '/home/xxx/runs/'
            os.makedirs(best_pred_dir, exist_ok=True)
            best_pred_image_path = os.path.join(best_pred_dir,
                                                str('pred{}.png'.format(epoch)))
            im.save(best_pred_image_path)
            logging.info(f'{epoch} saved!')

        if epoch_loss < maxx:
            torch.save(model.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            maxx = epoch_loss
            print("maxx", maxx)
            logging.info(f'Checkpoint {epoch} saved!')

    plt.plot(train_loss_list)
    plt.savefig('loss_fig/train_loss_SSIM.pdf')
    
    plt.clf()
    plt.plot(val_loss_list)
    plt.savefig('loss_fig/val_loss_SSIM.pdf')
    plt.close()
    


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--val_percent', '-v', dest='val_percent', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--output_channels', '-c', type=int, default=3, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda', 1)
    logging.info(f'Using device {device}')
    
    model = UNet(n_channels=3, output_channels=args.output_channels, bilinear=args.bilinear)
    model.to(device=device)
    train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val_percent
            )