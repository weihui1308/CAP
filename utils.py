import torch
import torch.nn as nn
import torchvision.transforms as TF
from torchvision.utils import save_image
import cv2
import numpy as np
import random
import kornia.geometry.transform as KT
import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTransformer(nn.Module):
    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.noise_factor = 0.1
        self.color_factor = 0.1
        self.ratio_h = 0.23
        self.ratio_w = 0.23
    
    
    def forward(self, patch, targets, imgs):
        patch_mask = torch.ones_like(patch).cuda()
        image_size = imgs.size()         # height, width

        patch_tmp = torch.zeros_like(imgs).cuda()
        patch_mask_tmp = torch.zeros_like(imgs).cuda()
        
        for i in range(targets.size(0)):
            img_idx = targets[i][0]
            bbox_w = targets[i][-2] * image_size[-1]
            bbox_h = targets[i][-1] * image_size[-2]
            
            # resize
            patch_width = int(bbox_h * self.ratio_w)
            patch_height= int(bbox_h * self.ratio_h)
            if patch_width == 0 or patch_height == 0:
                continue
            patch_size = (patch_height, patch_width)
            patch_resize = KT.resize(patch, patch_size)
            patch_mask_resize = KT.resize(patch_mask, patch_size)
            
            # rotation
            angle = random.uniform(-10, 10)
            patch_rotation = TF.functional.rotate(patch_resize, angle, expand=True)
            patch_mask_rotation = TF.functional.rotate(patch_mask_resize, angle, expand=True)
            
            patch_size_h = patch_rotation.size()[-1]
            patch_size_w = patch_rotation.size()[-2]
            
            # padding
            x_center = int(targets[i][2] * image_size[-1])
            y_center = int(targets[i][3] * image_size[-2])
            
            padding_h = image_size[-2] - patch_size_h
            padding_w = image_size[-1] - patch_size_w
            
            padding_left = x_center - int(0.5 * patch_size_w)
            padding_right = padding_w - padding_left
            
            padding_top = y_center - int(0.6 * patch_size_h)
            padding_bottom = padding_h - padding_top

            padding = nn.ZeroPad2d((int(padding_left), int(padding_right), int(padding_top), int(padding_bottom)))
            patch_padding = padding(patch_rotation)
            patch_mask_padding = padding(patch_mask_rotation)
            
            patch_tmp[int(img_idx.item())] += patch_padding.squeeze()
            patch_mask_tmp[int(img_idx.item())] += patch_mask_padding.squeeze()
            
        patch_tmp.data.clamp_(0,1)
        patch_mask_tmp.data.clamp_(0,1)
        
        return patch_tmp, patch_mask_tmp
        
        
        
class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, patch, patch_mask_tf):
        patch_mask = patch_mask_tf - 1
        patch_mask = - patch_mask

        img_batch = torch.mul(img_batch, patch_mask) + torch.mul(patch, patch_mask_tf)
        
        imgWithPatch = img_batch
        return imgWithPatch


# define draw
def plotCurve(x_vals, y_vals, 
                x_label, y_label, filename,
                legend=None,
                figsize=(10.0, 5.0)):
    # set figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals)
    
    if legend:
        plt.legend(legend)
    plt.savefig('results/'+filename)
    #plt.show()
    plt.close('all')

       
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
        
        
def image_loader(image_name, size):
    loader = TF.Compose([
          TF.Resize(size),  # scale imported image
          TF.CenterCrop(size),
          TF.ToTensor(),
          TF.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
          ])  # transform it into a torch tensor
    
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out
        
        
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G
    

# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename).convert('RGB')
    return img


# using ImageNet values
def normalize_tensor_transform():
    return TF.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def fitness_map50(x):
    w = [0.0, 0.0, 1.0, 0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)
        
        
        
        
        
        
        
        
        
        