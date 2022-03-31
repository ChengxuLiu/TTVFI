import sys
import os
import  threading
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import gradcheck
import numpy as np
import pdb

def charbonier_loss(x,epsilon):
    loss = torch.mean(torch.sqrt(x * x + epsilon * epsilon))
    return loss

def create_mask(tensor, paddings):
    bs,_,h,w = tensor.size()# bs * 1 * h * w
    inner_width = w - (paddings[0][0] + paddings[0][1])
    inner_height = h - (paddings[1][0] + paddings[1][1])
    inner = torch.ones(inner_height,inner_width).cuda() #h * w
    mask2d = F.pad(inner, (paddings[0][0],paddings[0][1],paddings[1][0],paddings[1][1]), "constant", 0)
    mask4d = mask2d.unsqueeze(0).unsqueeze(0).expand(bs,-1,-1,-1)
    return mask4d

def ternary_loss(im1, im2_warped, mask, epsilon, max_distance=3):
    patch_size = 2 * max_distance + 1
    def _ternary_transform(image):
        intensities=torch.autograd.Variable(torch.ones(image.size(0),1,image.size(2),image.size(3))).cuda()
        intensities=0.299*image[:,0,:,:]+0.587*image[:,1,:,:]+0.114*image[:,2,:,:]
        out_channels = patch_size * patch_size
        w = np.reshape(np.eye(out_channels,out_channels), (out_channels, 1, patch_size, patch_size))
        weights = torch.FloatTensor(w).cuda()
        patches = F.conv2d(torch.unsqueeze(intensities,1), weights, padding=max_distance)# out_channels * 1 * patch_size * patch_size
        transf = patches - torch.unsqueeze(intensities,1) # bs * patch_size * h * w
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = (t1 - t2)**2# bs * patch_size**2 * h * w
        dist_norm = dist / (0.1 + dist)# bs * patch_size**2 * h * w
        dist_sum = torch.sum(dist_norm, 1, keepdim=True)# bs * 1 * h * w
        return dist_sum

    t1 = _ternary_transform(im1)
    t2 = _ternary_transform(im2_warped)
    dist = _hamming_distance(t1, t2)
    mask = mask[:,0,:,:].unsqueeze(1)
    transform_mask = create_mask(mask, [[max_distance, max_distance],
                                        [max_distance, max_distance]])
    return charbonier_loss(dist * mask * transform_mask, epsilon)

def part_loss_s1(diffs, images, imgt, img01, epsilon):
    pixel_loss = [charbonier_loss(diff, epsilon) for diff in diffs]
    if imgt:
        output_1t,output_0t,mask_1t,mask_0t,cur_input_t = imgt[0],imgt[1],imgt[2],imgt[3],imgt[4]
        output_1t0,output_0t1,mask_1t0,mask_0t1 = img01[0],img01[1],img01[2],img01[3]
        cur_input_0, cur_input_1 = images[0],images[1]
        census_loss = ternary_loss(cur_input_t,output_0t,mask_0t, epsilon) \
                    + ternary_loss(cur_input_t,output_1t,mask_1t, epsilon) \
                    + ternary_loss(cur_input_0,output_1t0,mask_1t0, epsilon) \
                    + ternary_loss(cur_input_1,output_0t1,mask_0t1, epsilon)
    else:
        census_loss = 0.0
    return pixel_loss, census_loss



def part_loss_s2(diffs, epsilon):
    pixel_loss = [charbonier_loss(diff, epsilon) for diff in diffs]
    return pixel_loss

