import torch
import random

default_param = [
    0.4, 
    1024,
    10,
    0.4,
    256,
    200,
]

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

start = 0.4
end = 2.0

def get_random_param(device):
    param = default_param.copy()
    param[0] = random.uniform(start, end)
    param[1] = random.uniform(768, 1280)
    param[2] = random.uniform(4, 20)
    param[3] = random.uniform(0.4, 1.6)
    param[4] = random.uniform(200, 600)
    param[5] = random.uniform(200, 600)

    param = [
            normalize(param[0], 0.4, 2.0, 0, 1),
            normalize(param[1], 768, 1280, 0, 1),
            normalize(param[2], 4.0, 20.0, 0, 1),
            normalize(param[3], 0.4, 1.6, 0, 1),
            normalize(param[4], 200, 600, 0, 1),
            normalize(param[5], 200, 600, 0, 1),
    ]
    param = torch.Tensor(param).to(device=device).unsqueeze(0)
    return param


