import os
import time
st = time.time()
os.environ['MKL_THREADING_LAYER'] = 'GNU'

cmd = "CUDA_VISIBLE_DEVICES=2 python tester.py --uid 0002 --batch_size 8 --save_which 0 \
      --num_layer 2 --feat_channel 64 --patchsize 8 --n_head 4 \
      --pretrained ./checkpoint/TTVFI_stage2.pth \
      --datasetPath_vimeo ./dataset_triplet/Vimeo-90K/vimeo_triplet/ " 
print(cmd)
os.system(cmd)
end = time.time()
print('Time cost: ', (end - st)/3600, " hours")



