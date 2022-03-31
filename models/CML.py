
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from models.PWCNet.correlation_package_pytorch1_0.correlation import Correlation



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

import time

class CMLNet(nn.Module):
    def __init__(self, md=4):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(CMLNet,self).__init__()

        # self.conv0a  = conv(4,   16, kernel_size=3, stride=1)
        # self.conv0aa  = conv(16,   8, kernel_size=3, stride=1)
        # self.conv0b  = conv(8,   2, kernel_size=3, stride=1)

        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)

        self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*md+1)**2
        dd = np.cumsum([64, 64, 48, 32, 16],dtype=np.int32).astype(np.int)
        dd = [int(d) for d in dd]

        od = nd+32+2
        self.conv2_0 = conv(od,      64, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],64, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],48,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],32,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],16,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 
        # self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        # self.upfeat2 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)


        od = nd+16+2
        self.conv1_0 = conv(od,      64, kernel_size=3, stride=1)
        self.conv1_1 = conv(od+dd[0],64, kernel_size=3, stride=1)
        self.conv1_2 = conv(od+dd[1],48,  kernel_size=3, stride=1)
        self.conv1_3 = conv(od+dd[2],32,  kernel_size=3, stride=1)
        self.conv1_4 = conv(od+dd[3],16,  kernel_size=3, stride=1)
        self.predict_flow1 = predict_flow(od+dd[4]) 


        self.dc_conv1 = conv(od+dd[4], 64, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(64,      64, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(64,      64, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(64,      48,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(48,       32,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(32,       16,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def warp_with_mask(self, x, flo, return_mask=False):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        if return_mask:
            return output * mask, mask
        else:
            return output * mask


    def forward(self,im0,im1,flow01,flow10,timestep):
        # im0 #3*256*256
        # im1 #3*256*256
        # flow01 #2*64*64
        # flow10 #2*64*64
        c0_1 = self.conv1b(self.conv1aa(self.conv1a(im0)))#16*128*128
        c1_1 = self.conv1b(self.conv1aa(self.conv1a(im1)))#16*128*128
        c0_2 = self.conv2b(self.conv2aa(self.conv2a(c0_1)))#32*64*64
        c1_2 = self.conv2b(self.conv2aa(self.conv2a(c1_1)))#32*64*64
        
        flowt0_2 = (1-timestep)*timestep*(-flow01) + timestep*timestep*flow10#2*64*64
        flowt1_2 = (1-timestep)*(1-timestep)*flow01 + timestep*(1-timestep)*(-flow10)#2*64*64

        # flowt0_2 = self.conv0b(self.conv0aa(self.conv0a(torch.cat((-timestep*flow01,timestep*flow10),1))))#2*64*64
        # flowt1_2 = self.conv0b(self.conv0aa(self.conv0a(torch.cat(((1-timestep)*flow01,(timestep-1)*flow10),1))))#2*64*64

        # flowt0_2 = timestep*flow10#2*64*64
        # flowt1_2 = (1-timestep)*flow01#2*64*64

        # flowt0_2 = -timestep*flow01 #2*64*64
        # flowt1_2 = -(1-timestep)*flow10#2*64*64    
        

        c0t_2 = self.warp_with_mask(c0_2, flowt0_2*5.0)#32*64*64
        c1t_2 = self.warp_with_mask(c1_2, flowt1_2*5.0)#32*64*64

        corr10_2 = self.corr(c1t_2, c0t_2)#81*64*64
        corr10_2 = self.leakyRELU(corr10_2)
        x = torch.cat((corr10_2, c1t_2, flowt0_2), 1)#(81+32+2)*64*64
        x = torch.cat((self.conv2_0(x), x),1)#(81+32+2 +128)*64*64
        x = torch.cat((self.conv2_1(x), x),1)#(81+32+2 +128+128)*64*64
        x = torch.cat((self.conv2_2(x), x),1)#(81+32+2 +128+128+96)*64*64
        x = torch.cat((self.conv2_3(x), x),1)#(81+32+2 +128+128+96+64)*64*64
        x = torch.cat((self.conv2_4(x), x),1)#(81+32+2 +128+128+96+64+32)*64*64
        flowt0_2 = flowt0_2 + self.predict_flow2(x)#2*64*64
        up_flowt0_2 = F.interpolate(flowt0_2, scale_factor=2, mode='bilinear')

        corr01_2 = self.corr(c0t_2, c1t_2)#81*64*64
        corr01_2 = self.leakyRELU(corr01_2)
        x = torch.cat((corr01_2, c0t_2, flowt1_2), 1)#(81+32+2)*64*64
        x = torch.cat((self.conv2_0(x), x),1)#(81+32+2 +128)*64*64
        x = torch.cat((self.conv2_1(x), x),1)#(81+32+2 +128+128)*64*64
        x = torch.cat((self.conv2_2(x), x),1)#(81+32+2 +128+128+96)*64*64
        x = torch.cat((self.conv2_3(x), x),1)#(81+32+2 +128+128+96+64)*64*64
        x = torch.cat((self.conv2_4(x), x),1)#(81+32+2 +128+128+96+64+32)*64*64
        flowt1_2 = flowt1_2 + self.predict_flow2(x)#2*64*64
        up_flowt1_2 = F.interpolate(flowt1_2, scale_factor=2, mode='bilinear')


        c0t_1 = self.warp_with_mask(c0_1, up_flowt0_2*10.0)#16*128*128
        c1t_1 = self.warp_with_mask(c1_1, up_flowt1_2*10.0)#16*128*128

        corr10_1 = self.corr(c1t_1, c0t_1)#81*128*128
        corr10_1 = self.leakyRELU(corr10_1)
        x = torch.cat((corr10_1, c1t_1, up_flowt0_2), 1)#(81+16+2+2)*128*128
        x = torch.cat((self.conv1_0(x), x),1)#(81+16+2+2 +128)*128*128
        x = torch.cat((self.conv1_1(x), x),1)#(81+16+2+2 +128+128)*128*128
        x = torch.cat((self.conv1_2(x), x),1)#(81+16+2+2 +128+128+96)*128*128
        x = torch.cat((self.conv1_3(x), x),1)#(81+16+2+2 +128+128+96+64)*128*128
        x = torch.cat((self.conv1_4(x), x),1)#(81+16+2+2 +128+128+96+64+32)*128*128
        up_flowt0_2 = up_flowt0_2 + self.predict_flow1(x)#2*128*128
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        up_flowt0_2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))#2*128*128

        corr01_1 = self.corr(c0t_1, c1t_1)#81*128*128
        corr01_1 = self.leakyRELU(corr01_1)
        x = torch.cat((corr01_1, c0t_1, up_flowt1_2), 1)#(81+16+2+2)*128*128
        x = torch.cat((self.conv1_0(x), x),1)#(81+16+2+2 +128)*128*128
        x = torch.cat((self.conv1_1(x), x),1)#(81+16+2+2 +128+128)*128*128
        x = torch.cat((self.conv1_2(x), x),1)#(81+16+2+2 +128+128+96)*128*128
        x = torch.cat((self.conv1_3(x), x),1)#(81+16+2+2 +128+128+96+64)*128*128
        x = torch.cat((self.conv1_4(x), x),1)#(81+16+2+2 +128+128+96+64+32)*128*128
        up_flowt1_2 = up_flowt1_2 + self.predict_flow1(x)#2*128*128
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        up_flowt1_2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))#2*128*128

        return up_flowt0_2,up_flowt1_2


def cml_net(path=None):
    model = CMLNet()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'], strict=False)
            print("load CMLNet successful state_dict !!!")
        else:
            model.load_state_dict(data, strict=False)
            print("load CMLNet successful !!!")
    return model
