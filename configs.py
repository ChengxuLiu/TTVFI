import os
import datetime
import argparse
import numpy
import  torch

parser = argparse.ArgumentParser(description='TTVSR')

parser.add_argument('--datasetPath_vimeo',default='./Vimeo-90K/vimeo_triplet/',help = 'the path of datasets')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--numEpoch', '-e', type = int, default=100, help= 'Number of epochs to train(default:100)')
parser.add_argument('--batch_size', '-b',type = int ,default=4, help = 'batch size (default:4)' )
parser.add_argument('--iter_print', type=int,default=500, help ='print log (default:500)')
parser.add_argument('--num_layer', type=int,default=2, help ='Number of layer of Transformer (default:2)')
parser.add_argument('--feat_channel', type=int,default=64, help ='Number of channel of feature (default:64)')
parser.add_argument('--patchsize', type=int,default=8, help ='Size of shift window in attention (default:8)')
parser.add_argument('--n_head', type=int,default=4, help ='Number of head in multi-head attention (default:4)')

parser.add_argument('--save_which', '-s', type=int, default=0, help='choose which result to save: 0 ==> rectified')
parser.add_argument('--time_step',  type=float, default=0.5, help='choose the time stepfactors')
parser.add_argument('--flow_lr', type = float, default=0.00005, help = 'learning rate of motion estimation network (default: 0.00005)')
parser.add_argument('--pro_flow_lr', type = float, default=0.00005, help = 'learning rate of consistent motion learning to refine flow (default: 0.00005)')
parser.add_argument('--rectify_lr', type=float, default=0.0005, help  = 'the learning rate for trajectory-aware Transformer (default: 0.0005)')
parser.add_argument('--pixel_alpha', type=float, nargs='+', default=[0.5,0.5,0.5,0.5,0.5,1.0], help= 'the ration of loss for interpolated and rectified result (default: [0.0, 1.0])')
parser.add_argument('--census_alpha', type=float, default=0.0, help= 'the ration of census loss (default: 0.0)')

parser.add_argument('--epsilon', type = float, default=1e-6, help = 'the epsilon for charbonier loss,etc (default: 1e-6)')
parser.add_argument('--weight_decay', type = float, default=0, help = 'the weight decay for whole network ' )
parser.add_argument('--patience', type=int, default=5, help = 'the patience of reduce on plateou')
parser.add_argument('--epoch_eval', type=int, default=1, help = 'the number of evaluation during training')
parser.add_argument('--factor', type = float, default=0.2, help = 'the factor of reduce on plateou')

parser.add_argument('--pretrained', dest='SAVED_MODEL', default=None, help ='path to the pretrained model weights')
parser.add_argument('--use_cuda', default= True, type = bool, help='use cuda or not')
parser.add_argument('--use_cudnn',default=1,type=int, help = 'use cudnn or not')
parser.add_argument('--dtype', default=torch.cuda.FloatTensor, choices = [torch.cuda.FloatTensor,torch.FloatTensor],help = 'tensor data type ')
parser.add_argument('--uid', type=str, default= None, help='unique id for the job')
parser.add_argument('--save_path', type = str, default = './', help = 'the output dir of weights')
args = parser.parse_args()

import shutil

if args.uid == None:
    unique_id = str(numpy.random.randint(0, 100000))
    print("revise the unique id to a random numer " + str(unique_id))
    args.uid = unique_id
    timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H-%M")
    save_path = args.save_path + 'model_weights/'+ args.uid  +'-' + timestamp
else:
    save_path = args.save_path + 'model_weights/'+ str(args.uid)

# print("no pth here : " + save_path + "/best"+".pth")
if not os.path.exists(save_path + "/best"+".pth"):
    os.makedirs(save_path,exist_ok=True)

parser.add_argument('--log', default = save_path+'/log.txt', help = 'the log file in training')
parser.add_argument('--arg', default = save_path+'/args.txt', help = 'the args used')

args = parser.parse_args()


with open(args.log, 'w') as f:
    f.close()
with open(args.arg, 'w') as f:
    print(args)
    print(args,file=f)
    f.close()
if args.use_cudnn:
    print("cudnn is used")
    torch.backends.cudnn.benchmark = True  # to speed up the
else:
    print("cudnn is not used")
    torch.backends.cudnn.benchmark = False  # to speed up the

