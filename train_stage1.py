import os
import time
st = time.time()
os.environ['MKL_THREADING_LAYER'] = 'GNU'

cmd = "CUDA_VISIBLE_DEVICES=0 python trainer_stage1.py --uid 0000 --batch_size 4 --flow_lr 0.00005 --pro_flow_lr 0.00005 --numEpoch 20 \
      --pixel_alpha 0.0 1.0 1.0 1.0 1.0 1.0 --census_alpha 1.0 --patience 4 --factor 0.2 --iter_print 500  \
      --datasetPath ./Vimeo-90K/vimeo_triplet/  "

print(cmd)
os.system(cmd)
end = time.time()
print('Time cost: ', (end - st)/3600, " hours")



