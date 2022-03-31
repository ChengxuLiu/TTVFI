import os
import time
st = time.time()
os.environ['MKL_THREADING_LAYER'] = 'GNU'

cmd = "CUDA_VISIBLE_DEVICES=0 python trainer_stage2.py --uid 0001 --batch_size 4 --flow_lr 0.00005 --pro_flow_lr 0.00005 --rectify_lr 0.0005  --numEpoch 70\
      --pixel_alpha 1.0 0.0 0.0 0.0 0.0 0.0 --census_alpha 0.0 --patience 4 --factor 0.2 --iter_print 500 \
      --num_layer 2 --feat_channel 64 --patchsize 8 --n_head 4 \
      --pretrained ./checkpoint/TTVFI_stage1.pth \
      --datasetPath_vimeo ./Vimeo-90K/vimeo_triplet/  "

print(cmd)
os.system(cmd)
end = time.time()
print('Time cost: ', (end - st)/3600, " hours")
