# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PWCNet.PWCNet import pwc_dc_net
from models.Transformer import VFIT
from models.CML import cml_net
import time
import pdb


class TTVFI(torch.nn.Module):
    def __init__(self,
                 num_layer=2,
                 feat_channel=32,
                 patchsize=8,
                 n_head=2,
                 timestep=0.5,
                 training=True):

        # base class initialization
        super(TTVFI, self).__init__()
        
        self.training = training
        self.timestep = timestep
        self.numFrames =int(1.0/timestep) - 1
        self.cml = cml_net()
        self.rectifyNet = VFIT(num_layer=num_layer, feat_channel=feat_channel, patchsize=patchsize, n_head=n_head,timestep = self.timestep)
        self._initialize_weights()
        if self.training:
            self.flownets = pwc_dc_net(path = "./models/PWCNet/pwc_net.pth.tar")
        else:
            self.flownets = pwc_dc_net()
        self.div_flow = 20.0
        return

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):

        """
        Parameters
        ----------
        input: shape (3, batch, 3, width, height)
        -----------
        """
        losses = []
        offsets= []
 
        if self.training == True:
            assert input.size(0) == 3
            input_0,input_t,input_1 = torch.squeeze(input,dim=0)
        else:
            assert input.size(0) ==2
            input_0,input_1 = torch.squeeze(input,dim=0)

        #prepare the input data of current scale
        cur_input_0 = input_0
        if self.training == True:
            cur_input_t = input_t
        cur_input_1 =  input_1

        Flow01 = self.flownets(torch.cat((cur_input_0, cur_input_1), dim=1),output_more=False) # bs * 2 * 64 * 64   (bs,2,h,w)
        Flow10 = self.flownets(torch.cat((cur_input_1, cur_input_0), dim=1),output_more=False)

#################################################
        Flowt0_s, Flowt1_s = self.cml(cur_input_0,cur_input_1,Flow01,Flow10,self.timestep)
        Flowt0_s = F.interpolate(Flowt0_s, scale_factor=2, mode='bilinear') * self.div_flow
        Flowt1_s = F.interpolate(Flowt1_s, scale_factor=2, mode='bilinear') * self.div_flow
#################################################
#################################################
        Flow01 = F.interpolate(Flow01, scale_factor=4, mode='bilinear') * self.div_flow
        Flow10 = F.interpolate(Flow10, scale_factor=4, mode='bilinear') * self.div_flow
        Flowt0 = (1-self.timestep)*self.timestep*(-Flow01) + self.timestep*self.timestep*Flow10#2*64*64
        Flowt1 = (1-self.timestep)*(1-self.timestep)*Flow01 + self.timestep*(1-self.timestep)*(-Flow10)#2*64*64
#################################################
        rectify_frame = torch.cat((cur_input_0,cur_input_1),dim =1)
        rectify_flow = torch.cat((Flowt0,Flowt1),dim =1)
        rectify_flow_s = torch.cat((Flowt0_s,Flowt1_s),dim =1)

        cur_output_rectified = self.rectifyNet(rectify_frame,rectify_flow,rectify_flow_s)

        if self.training == True:
            losses += [cur_output_rectified - cur_input_t] 
        if self.training == True:
            return losses, cur_output_rectified
        else:
            return cur_output_rectified


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

    def warp_no_mask(self, x, flo):
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
        return output



