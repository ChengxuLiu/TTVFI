# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PWCNet.PWCNet import pwc_dc_net
from models.CML import cml_net
import time
import pdb


class CML(torch.nn.Module):
    def __init__(self,
                 timestep=0.5,
                 training=True):

        # base class initialization
        super(CML, self).__init__()
        self.training = training
        self.timestep = timestep
        assert (timestep == 0.5) # TODO: or else the WeigtedFlowProjection should also be revised... Really Tedious work.
        self.numFrames =int(1.0/timestep) - 1
        self._initialize_weights()
        self.cml = cml_net()
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
        Flowt0, Flowt1 = self.cml(cur_input_0,cur_input_1,Flow01,Flow10,self.timestep)

        Flowt0 = F.interpolate(Flowt0, scale_factor=2, mode='bilinear') * self.div_flow
        Flowt1 = F.interpolate(Flowt1, scale_factor=2, mode='bilinear') * self.div_flow
        output_1t,mask_1t = self.warp_with_mask(cur_input_1, Flowt1)   # F_t_0
        output_0t,mask_0t = self.warp_with_mask(cur_input_0, Flowt0)   # F_t_0

        output_t = (output_0t + output_1t) / 2
        cur_output_rectified = output_t + 0.5 * (output_1t * (1-mask_0t) + output_0t * (1-mask_1t))

        Flow01 = F.interpolate(Flow01, scale_factor=4, mode='bilinear') * self.div_flow
        Flow10 = F.interpolate(Flow10, scale_factor=4, mode='bilinear') * self.div_flow
        Flow0t = self.timestep * Flow01
        Flow1t = self.timestep * Flow10
        output_1t0,mask_1t0 = self.warp_with_mask(output_1t, Flow0t)
        output_0t1,mask_0t1 = self.warp_with_mask(output_0t, Flow1t)


        if self.training == True:
            losses += [cur_output_rectified - cur_input_t] 
            losses += [output_1t - output_0t]
            losses += [(cur_input_0 - output_1t0)*mask_1t0]
            losses += [(cur_input_1 - output_0t1)*mask_0t1]
            losses += [cur_input_t - output_1t]
            losses += [cur_input_t - output_0t]
            offsets +=[[Flowt0, Flowt1]]

        return losses, [output_1t,output_0t,mask_1t,mask_0t, cur_input_t], [output_1t0,output_0t1,mask_1t0,mask_0t1]


    def warp_with_mask(self, x, flo, return_mask=True):
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


    def forward_flownets(self, model, input, time_offsets = None):

        if time_offsets == None :
            time_offsets = [0.5]
        elif type(time_offsets) == float:
            time_offsets = [time_offsets]
        elif type(time_offsets) == list:
            pass
        temp = model(input,output_more=True)  # this is a single direction motion results, but not a bidirectional one
        temps = [self.div_flow * temp[0] * time_offset for time_offset in time_offsets]# single direction to bidirection should haven it.
        temps = [nn.Upsample(scale_factor=4, mode='bilinear')(temp)  for temp in temps]# nearest interpolation won't be better i think
        return temps



