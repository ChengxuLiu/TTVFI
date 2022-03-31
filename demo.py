import sys
import time
import os
from torch.autograd import Variable
import math
import torch
import random
import numpy as np
import numpy
from models.TTVFI_S2 import TTVFI
from configs import args
from scipy.misc import imread, imsave
import warnings
warnings.filterwarnings("ignore")
# device = torch.device('cuda' if args.cuda else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


args.SAVED_MODEL = "./checkpoint/TTVFI_stage2.pth"
First = "./im1.png"
Second = "./im3.png"
Out = "./output.png"


model = TTVFI(num_layer=2,feat_channel=64,patchsize=8,n_head=4,timestep=0.5,training=False)
if args.use_cuda:
    model = model.cuda()
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")
model = model.eval() # deploy mode

X0 =  torch.from_numpy( np.transpose(imread(First) , (2,0,1)).astype("float32")/ 255.0).type(args.dtype)
X1 =  torch.from_numpy( np.transpose(imread(Second) , (2,0,1)).astype("float32")/ 255.0).type(args.dtype)

assert (X0.size(1) == X1.size(1))
assert (X0.size(2) == X1.size(2))

intWidth = X0.size(2)
intHeight = X0.size(1)
channel = X0.size(0)
if not channel == 3:
    continue

if intWidth != ((intWidth >> 7) << 7):
    intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
    intPaddingLeft =int(( intWidth_pad - intWidth)/2)
    intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
else:
    intWidth_pad = intWidth
    intPaddingLeft = 0
    intPaddingRight= 0

if intHeight != ((intHeight >> 7) << 7):
    intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
    intPaddingTop = int((intHeight_pad - intHeight) / 2)
    intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
else:
    intHeight_pad = intHeight
    intPaddingTop = 0
    intPaddingBottom = 0

pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

torch.set_grad_enabled(False)
X0 = Variable(torch.unsqueeze(X0,0))
X1 = Variable(torch.unsqueeze(X1,0))
X0 = pader(X0)
X1 = pader(X1)

if args.use_cuda:
    X0 = X0.cuda()
    X1 = X1.cuda()

output = model(torch.stack((X0,X1),dim = 0))

if args.use_cuda:
    output = output.data.cpu().numpy()
else:
    output = output.data.numpy()
output = np.transpose(255.0 * output.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
imsave(Out, np.round(output).astype(numpy.uint8))