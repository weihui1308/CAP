from .unet_parts import *

def normalize(x, a, b, c, d):
    '''
    x from (a, b) to (c, d)
    '''
    return (float(x) - a) * (float(d) - c) / (float(b) - a) + float(c)

class UNet(nn.Module):
    def __init__(self, n_channels, output_channels, bilinear=False, img_size=300):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.output_channels = output_channels
        self.bilinear = bilinear
        self.img_size = img_size

        # the added channels
        param_num = 6

        self.Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.inc = (DoubleConv(n_channels + param_num, 64))
        self.down1 = (Down(64 + param_num, 128))
        self.down2 = (Down(128 + param_num, 256))
        self.down3 = (Down(256 + param_num, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512 + param_num, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, output_channels))

        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x, param_list=None):

        param_list = param_list.view(param_list.size(0), param_list.size(1), 1, 1)
        param_list = param_list.repeat(1, 1, x.size(2), x.size(3))
        param_layer = param_list

        x1 = self.inc(torch.cat([x, param_layer], dim=1))
        x2 = self.down1(torch.cat([x1, param_layer], dim=1))
        x3 = self.down2(torch.cat([x2, self.Avgpool(param_layer)], dim=1))
        x4 = self.down3(torch.cat([x3, self.Avgpool(self.Avgpool(param_layer))], dim=1))
        x5 = self.down4(torch.cat([x4, self.Avgpool(self.Avgpool(self.Avgpool(param_layer)))], dim=1))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

