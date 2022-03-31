import sys
import os
import cv2
import lpips
import threading
import torch
import time
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
import numpy
from math import exp
from losses.AverageMeter import AverageMeter
from losses.loss_function import part_loss_s2
from losses.lr_scheduler import ReduceLROnPlateau
from data.datasets import Vimeo_90K_loader
from data.balancedsampler import RandomBalancedSampler
from models.TTVFI_S2 import TTVFI
from configs import args
import warnings
warnings.filterwarnings("ignore")


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().cuda()
    return window


def calculate_ssim3(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret



def train():
    torch.manual_seed(args.seed)

    model = TTVFI(num_layer=args.num_layer,feat_channel=args.feat_channel,patchsize=args.patchsize,n_head=args.n_head,timestep=args.time_step,training=True)
    if args.use_cuda:
        print("Turn the model into CUDA")
        model = model.cuda()
    if not args.SAVED_MODEL==None:
        print("Fine tuning on " +  args.SAVED_MODEL)
        if not  args.use_cuda:
            pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(args.SAVED_MODEL)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        pretrained_dict = None

    vimeo_test_set = Vimeo_90K_loader(data_root=args.datasetPath,is_training=False)
    vimeo_val_loader = torch.utils.data.DataLoader(vimeo_test_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=1, pin_memory=True if args.use_cuda else False)

    print('{} vimeo test samples '.format(len(vimeo_test_set)))

    def count_network_parameters(model):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([numpy.prod(p.size()) for p in parameters])
        return N
    print("Num. of model parameters is :" + str(count_network_parameters(model)))
    
    val_total_PSNR_loss = AverageMeter()
    val_total_SSIM_loss = AverageMeter()
    start = time.perf_counter()
    for i, (img0, gt, img1) in enumerate(vimeo_val_loader):
        if i >=  int(len(vimeo_test_set)/ args.batch_size):
            break
        with torch.no_grad():
            img0 = img0.cuda() if args.use_cuda else img0
            gt = gt.cuda() if args.use_cuda else gt
            img1 = img1.cuda() if args.use_cuda else img1

            diffs, output = model(torch.stack((img0,gt,img1),dim = 0))
            per_sample_pix_error = torch.mean(torch.mean(torch.mean(diffs[args.save_which] ** 2,dim=1),dim=1),dim=1)
            per_sample_pix_error = per_sample_pix_error.data # extract tensor
            psnr_loss = torch.mean(20 * torch.log10(1.0/torch.sqrt(per_sample_pix_error)))
            val_total_PSNR_loss.update(psnr_loss.item(),args.batch_size)

            ssim = calculate_ssim3(output, gt, val_range=1.)
            val_total_SSIM_loss.update(ssim.item(),args.batch_size)
            print(".",end='',flush=True)
    print(
          "\tValidate PSNR: " + str([round(float(val_total_PSNR_loss.avg), 5)]) +
          "\tValidate SSIM: " + str([round(float(val_total_SSIM_loss.avg), 5)])
          )

    print("*********Finish Testing********")

if __name__ == '__main__':
    sys.setrecursionlimit(100000)# 0xC00000FD exception for the recursive detach of gradients.
    threading.stack_size(200000000)# 0xC00000FD exception for the recursive detach of gradients.
    thread = threading.Thread(target=train)
    thread.start()
    thread.join()

    exit(0)
