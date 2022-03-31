import sys
import os
import cv2
import threading
import torch
import time
from torch.autograd import Variable
import torch.utils.data
import numpy
from losses.AverageMeter import AverageMeter
from losses.loss_function import part_loss_s2
from losses.lr_scheduler import ReduceLROnPlateau
from data.datasets import Vimeo_90K_loader
from data.balancedsampler import RandomBalancedSampler
from models.TTVFI_S2 import TTVFI
from configs import args

import warnings
warnings.filterwarnings("ignore")

def train():
    torch.manual_seed(args.seed)

    model = TTVFI(num_layer=args.num_layer,feat_channel=args.feat_channel,patchsize=args.patchsize,n_head=args.n_head,timestep=args.time_step,training=True)
    if args.use_cuda:
        print("Turn the model into CUDA")
        model = model.cuda()
    args.save_path =args.save_path+'weights/'+ str(args.uid)
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

    vimeo_train_set = Vimeo_90K_loader(data_root=args.datasetPath_vimeo,is_training=True)
    vimeo_test_set = Vimeo_90K_loader(data_root=args.datasetPath_vimeo,is_training=False)
    vimeo_train_loader = torch.utils.data.DataLoader(vimeo_train_set, batch_size = args.batch_size,
         sampler=RandomBalancedSampler(vimeo_train_set, int(len(vimeo_train_set) / args.batch_size )),
                                             num_workers=4, pin_memory=True if args.use_cuda else False)
    vimeo_val_loader = torch.utils.data.DataLoader(vimeo_test_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=2, pin_memory=True if args.use_cuda else False)

    print('{} vimeo train samples '.format(len(vimeo_train_set)))
    print('{} vimeo test samples '.format(len(vimeo_test_set)))
    
    print("train the interpolation net")
    optimizer = torch.optim.Adamax([
                {'params': model.flownets.parameters(), 'lr': args.flow_lr},
                {'params': model.cml.parameters(), 'lr': args.pro_flow_lr},
                {'params': model.rectifyNet.parameters(), 'lr': args.rectify_lr}
            ], betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)


    scheduler = ReduceLROnPlateau(optimizer, 'min' ,factor=args.factor, patience=args.patience, verbose=True)
    print("*********Start Training********")
    print("Flow lr is: "+ str(float(optimizer.param_groups[0]['lr'])))
    print("Refine Flow lr is: "+ str(float(optimizer.param_groups[1]['lr'])))
    print("Reconstruction lr is: "+ str(float(optimizer.param_groups[2]['lr'])))
    print("Iter/Epoch is: "+ str(int(len(vimeo_train_set) / args.batch_size )))
    print("Num of EPOCH is: "+ str(args.numEpoch))
    def count_network_parameters(model):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([numpy.prod(p.size()) for p in parameters])
        return N
    print("Num. of model parameters is :" + str(count_network_parameters(model)))
    
    training_losses = AverageMeter()
    val_total_losses = AverageMeter()
    val_total_pixel_loss = AverageMeter()
    val_total_PSNR_loss = AverageMeter()

    auxiliary_data = []
    saved_total_loss = 10e10
    saved_total_PSNR = -1
    lr_flow = optimizer.param_groups[0]['lr']
    lr_cml = optimizer.param_groups[1]['lr']
    lr_recon = optimizer.param_groups[2]['lr']
    start = time.perf_counter()

    for t in range(args.numEpoch):
        print("The id of this in-training network is " + str(args.uid))
        #Turn into training mode
        model = model.train()
        for i, (img0, gt, img1) in enumerate(vimeo_train_loader):
            if i >= int(len(vimeo_train_set) / args.batch_size):
                break
            img0 = img0.cuda() if args.use_cuda else img0
            gt = gt.cuda() if args.use_cuda else gt
            img1 = img1.cuda() if args.use_cuda else img1
            img0 = Variable(img0, requires_grad= False)
            gt = Variable(gt, requires_grad= False)
            img1  = Variable(img1,requires_grad= False)

            diffs, output = model(torch.stack((img0,gt,img1),dim = 0))
            pixel_loss = part_loss_s2(diffs,epsilon=args.epsilon)
            total_pixel_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.pixel_alpha, pixel_loss))
            total_loss = total_pixel_loss
            training_losses.update(total_loss.item(), args.batch_size)
            if i % args.iter_print == 0:
                end = time.perf_counter()
                print("Ep [" + str(t) +"/" + str(i) +
                                    "]\tTime: " + str(round(float((end-start)/args.iter_print),4))+
                                    "\tlr_flow: " + str(round(float(lr_flow),7))+
                                    "\tlr_cml: " + str(round(float(lr_cml),7)) +
                                    "\tlr_recon: " + str(round(float(lr_recon),7)) +
                                    "\tTotal: " + str([round(x.item(),5) for x in [total_loss]]) +
                                    "\tAvg. Loss: " + str([round(training_losses.avg, 5)]))
                start = time.perf_counter()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if t == 1:
            # delete the pre validation weights for cleaner workspace
            if os.path.exists(args.save_path + "/epoch" + str(0) +".pth" ):
                os.remove(args.save_path + "/epoch" + str(0) +".pth")

        if os.path.exists(args.save_path + "/epoch" + str(t-1) +".pth"):
            os.remove(args.save_path + "/epoch" + str(t-1) +".pth")
        torch.save(model.state_dict(), args.save_path + "/epoch" + str(t) +".pth")


        print("\t\t**************Start Validation*****************")
        # Turn into evaluation mode
        if (t+1)%args.epoch_eval==0:
            auxiliary_data = []
            for i, (img0, gt, img1) in enumerate(vimeo_val_loader):
                if i >=  int(len(vimeo_test_set)/ args.batch_size):
                    break
                with torch.no_grad():
                    img0 = img0.cuda() if args.use_cuda else img0
                    gt = gt.cuda() if args.use_cuda else gt
                    img1 = img1.cuda() if args.use_cuda else img1

                    diffs, output = model(torch.stack((img0,gt,img1),dim = 0))
                    pixel_loss = part_loss_s2(diffs,epsilon=args.epsilon)
                    val_pixel_loss = sum(x * y for x, y in zip(args.pixel_alpha, pixel_loss))
                    val_total_loss = val_pixel_loss
                    per_sample_pix_error = torch.mean(torch.mean(torch.mean(diffs[args.save_which] ** 2,dim=1),dim=1),dim=1)
                    per_sample_pix_error = per_sample_pix_error.data # extract tensor
                    psnr_loss = torch.mean(20 * torch.log10(1.0/torch.sqrt(per_sample_pix_error)))
                    val_total_losses.update(val_total_loss.item(),args.batch_size)
                    val_total_pixel_loss.update(pixel_loss[args.save_which].item(), args.batch_size)
                    val_total_PSNR_loss.update(psnr_loss.item(),args.batch_size)
                    print(".",end='',flush=True)

            print("\nEpoch " + str(int(t)) +
                  "\tlearning rate flow: " + str(float(lr_flow)) +
                  "\tlearning rate refine flow: " + str(float(lr_cml)) +
                  "\tlearning rate reconstruction: " + str(float(lr_recon)) +
                  "\tAvg Training Loss: " + str(round(training_losses.avg,5)) +
                  "\tValidate Loss: " + str([round(float(val_total_losses.avg), 5)]) +
                  "\tValidate PSNR: " + str([round(float(val_total_PSNR_loss.avg), 5)]) +
                  "\tPixel Loss: " + str([round(float(val_total_pixel_loss.avg), 5)])
                  )
            auxiliary_data.append(['Epoch:',t, 'PSNR:', val_total_PSNR_loss.avg, 'Vimeo', 
                                       training_losses.avg, val_total_losses.avg, val_total_pixel_loss.avg])
            file = open(args.log,'a')
            file.write(str(auxiliary_data)+'\n')
            file.close()



            print("\t\tFinished an epoch, Check and Save the model weights")
                # we check the validation loss instead of training loss. OK~
            if saved_total_loss >= val_total_losses.avg:
                saved_total_loss = val_total_losses.avg
                torch.save(model.state_dict(), args.save_path + "/best"+".pth")
                print("\t\tBest Weights updated for decreased validation loss\n")

            else:
                print("\t\tWeights Not updated for undecreased validation loss\n")

            #schdule the learning rate
            scheduler.step(val_total_losses.avg)
            lr_flow = optimizer.param_groups[0]['lr']
            lr_cml = optimizer.param_groups[1]['lr']
            lr_recon = optimizer.param_groups[2]['lr']
            val_total_pixel_loss.reset()
            val_total_PSNR_loss.reset()
        training_losses.reset()



    print("*********Finish Training********")

if __name__ == '__main__':
    sys.setrecursionlimit(100000)# 0xC00000FD exception for the recursive detach of gradients.
    threading.stack_size(200000000)# 0xC00000FD exception for the recursive detach of gradients.
    thread = threading.Thread(target=train)
    thread.start()
    thread.join()

    exit(0)
