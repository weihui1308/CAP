import torch.nn.functional as F
import logging
import torch
import argparse
from PIL import Image
from torchvision.transforms import transforms
from torchvision import utils as vutils
from dataloader import BasicDataset
from unet import UNet
import os


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

def normalize(x, a, b, c, d):
    '''
    x from (a, b) to (c, d)
    '''
    return (float(x) - a) * (float(d) - c) / (float(b) - a) + float(c)

def get_args():
    parser = argparse.ArgumentParser(description='Predict ISP target from input images')
    parser.add_argument('--model_path', default=r'/home/xxx/checkpoint.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--img_path', default=r'/home/xxx/orig_image/')
    parser.add_argument('--param_path', default=r'/home/xxx/PARAM/')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    net = UNet(n_channels=3, output_channels=3, bilinear=args.bilinear)
    
    device = torch.device('cuda', 0)
    net.to(device=device)

    state_dict = torch.load(args.model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device=device)
    net.eval()
    logging.info('Model loaded!')
    
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
    ])


    img_paths = []
    img_path = args.img_path
    for file_name in os.listdir(img_path):
        if file_name.lower().endswith((".png")):
            img_paths.append(os.path.join(img_path, file_name))

    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.to(device=device)
        img = img.unsqueeze(0)

        file_name = os.path.basename(img_path)
        param_file_name = os.path.splitext(file_name)[0]+'.conf'
        param_path= os.path.join(args.param_path, param_file_name)
        with open(param_path, 'r') as f:
            lines_ = f.readlines()
        param = [
            normalize(float(lines_[0]), 0.4, 2, 0, 1),
            normalize(float(lines_[1]), 768, 1280, 0, 1),
            normalize(float(lines_[2]), 4.0, 20.0, 0, 1),
            normalize(float(lines_[3]), 0.4, 1.6, 0, 1),
            normalize(float(lines_[4]), 200, 600, 0, 1),
            normalize(float(lines_[5]), 200, 600, 0, 1),
        ]

        param = torch.tensor(param)
        param = param.to(device=device)
        param = param.unsqueeze(0)
        pred = net(img, param)
        img_save = pred[0].detach()
        im = transforms.ToPILImage()(img_save)
        output_dir = "/home/xxx/output/"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, file_name)
        
        print("save_path=", save_path)
        im.save(save_path)