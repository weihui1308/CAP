import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import pytorch_ssim


@torch.inference_mode()
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0
    
    criterion = pytorch_ssim.SSIM()

    # iterate over the validation set

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        orig_image, target_img, param = batch['orig_image'], batch['target_img'], batch['param']
        orig_image = orig_image.to(device=device, dtype=torch.float32)
        target_img = target_img.to(device=device, dtype=torch.float32)
        param = param.to(device=device, dtype=torch.float32)

        pred = net(orig_image, param)
        loss = 1 - criterion(target_img, pred)
        val_loss += loss.detach().cpu().item()
        
    return val_loss / num_val_batches