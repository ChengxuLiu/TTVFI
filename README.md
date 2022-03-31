# TTVFI
This is the official PyTorch implementation of the paper [Learning Trajectory-Aware Transformer for Video Frame Interpolation](ArxivLink).

## Contents
- [Introduction](#introduction)
  - [Contribution](#contribution)
  - [Overview](#overview)
  - [Visual](#Visual)
- [Requirements and dependencies](#requirements-and-dependencies)
- [Model and Results](#model-and-results)
- [Dataset](#dataset)
- [Demo](#demo)
- [Test](#test)
- [Train](#train)
- [Citation](#citation)
- [Contact](#contact)


## Introduction
<!-- We proposed an approach named TTVFI to study video super-resolution by leveraging long-range frame dependencies. TTVSR introduces Transformer architectures in video super-resolution tasks and formulates video frames into pre-aligned trajectories of visual tokens to calculate attention along trajectories. -->
<img src="./fig/teaser_TT-VFI.png" width=70%>

### Contribution
<!-- We propose a novel trajectory-aware Transformer, which is one of the first works to introduce Transformer into video super-resolution tasks. TTVSR reduces computational costs and enables long-range modeling in videos. TTVSR can outperform existing SOTA methods in four widely-used VSR benchmarks.
 -->

### Overview
<img src="./fig/framework_TT-VFI.png" width=100%>

### Visual
<img src="./fig/result_TT-VFI.png" width=90%>

## Requirements and dependencies
* python 3.6 (recommend to use [Anaconda](https://www.anaconda.com/))
* pytorch == 1.2.0
* torchvision == 0.4.0
* opencv-python == 4.5.5
* scikit-image == 0.17.2
* scipy == 1.1.0
* setuptools == 58.0.4
* Pillow == 8.4.0
* imageio == 2.15.0
* numpy == 1.19.5

## Model and Results
Pre-trained models can be downloaded from [onedrive](https://1drv.ms/u/s!Au4fJlmAZDhlhwjmP0D2RJOQaFqF?e=UHVz3H), [google drive](https://drive.google.com/drive/folders/1JWl22XUc0IOp1mx79_DRtwOwHjO1FP8I?usp=sharing), and [baidu cloud](https://pan.baidu.com/s/1nCjVhwArNajWFDDYwt4IUA)(j3nd).
* *TTVSR_stage1.pth*: trained from first stage with consistent motion learning.
* *TTVSR_stage2.pth*: trained from second stage with trajectory-aware Transformer on Viemo-90K dataset.

The output results on Vimeo-90K testing set, DAVIS, UCF101 and SNU-FILM can be downloaded from [onedrive](https://1drv.ms/u/s!Au4fJlmAZDhlhwjmP0D2RJOQaFqF?e=UHVz3H), [google drive](https://drive.google.com/drive/folders/1JWl22XUc0IOp1mx79_DRtwOwHjO1FP8I?usp=sharing), and [baidu cloud](https://pan.baidu.com/s/1nCjVhwArNajWFDDYwt4IUA)(j3nd).


## Dataset
1. Training set
	* [Viemo-90K](https://github.com/anchen1011/toflow) dataset. Download the [both triplet training and test set](http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip). The `tri_trainlist.txt` file listing the training samples in the download zip file.
		- Make Vimeo-90K structure be:
		```
			├────vimeo_triplet
				├────sequences
					├────00001
					├────...
					├────00078
				├────tri_trainlist.txt
				├────tri_testlist.txt
        ```

2. Testing set
    * [Viemo-90K](https://github.com/anchen1011/toflow) testset. The `tri_testlist.txt` file listing the testing samples in the download zip file.
    * [DAVIS](https://github.com/HyeongminLEE/AdaCoF-pytorch/tree/master/test_input/davis), [UCF101](https://drive.google.com/file/d/0B7EVK8r0v71pdHBNdXB6TE1wSTQ/view?resourcekey=0-r6ihCy20h3kbgZ3ZdimPiA), and [SNU-FILM](https://myungsub.github.io/CAIN/) dataset.
		- Make DAVIS, UCF101, and SNU-FILM structure be:
		```
			├────DAVIS
				├────input
				├────gt
			├────UCF101
				├────1
				├────...
			├────SNU-FILM
				├────test
					├────GOPRO_test
					├────YouTube_test
				├────test-easy.txt			
				├────...		
				├────test-extreme.txt		
        ```
## Demo
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/TTVFI.git
cd TTVFI
```
2. Generate the Correlation package required by [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master):
```
cd ./models/PWCNet/correlation_package_pytorch1_0/
./build.sh
```
3. Download pre-trained weights ([onedrive](https://1drv.ms/u/s!Au4fJlmAZDhlhwjmP0D2RJOQaFqF?e=UHVz3H)|[google drive](https://drive.google.com/drive/folders/1JWl22XUc0IOp1mx79_DRtwOwHjO1FP8I?usp=sharing)|[baidu cloud](https://pan.baidu.com/s/1nCjVhwArNajWFDDYwt4IUA)(j3nd)) under `./checkpoint`
```
cd ../../..
mkdir checkpoint
```
4. Prepare input frames and modify "FirstPath" and "SecondPath" in `./demo.py`
5. Run demo
```
python demo.py
```



## Test
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/TTVFI.git
cd TTVFI
```
2. Generate the Correlation package required by [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master):
```
cd ./models/PWCNet/correlation_package_pytorch1_0/
./build.sh
```
3. Download pre-trained weights ([onedrive](https://1drv.ms/u/s!Au4fJlmAZDhlhwjmP0D2RJOQaFqF?e=UHVz3H)|[google drive](https://drive.google.com/drive/folders/1JWl22XUc0IOp1mx79_DRtwOwHjO1FP8I?usp=sharing)|[baidu cloud](https://pan.baidu.com/s/1nCjVhwArNajWFDDYwt4IUA)(j3nd)) under `./checkpoint`
```
cd ../../..
mkdir checkpoint
```
4. Prepare testing dataset and modify "datasetPath" in `./test.py`
5. Run test
```
mkdir weights
# Vimeo
python test.py
```

## Train
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/TTVFI.git
cd TTVFI
```
2. Generate the Correlation package required by [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master):
```
cd ./models/PWCNet/correlation_package_pytorch1_0/
./build.sh
```
3. Prepare training dataset and modify "datasetPath" in `./train_stage1.py` and `./train_stage2.py`
4. Run training of stage1
```
mkdir weights
# stage one
python train_stage1.py
```
5. The models of stage1 are saved in `./weights` and fed into stage2 (modify "pretrained" in `./train_stage2.py`)
6. Run training of stage2
```
# stage two
python train_stage2.py
```
7. The models of stage2 are also saved in `./weights`


## Citation
<!-- If you find the code and pre-trained models useful for your research, please consider citing our paper. :blush:
```
@InProceedings{liu2022learning,
author = {Liu, Chengxu and Yang, Huan and Fu, Jianlong and Qian, Xueming},
title = {Learning Trajectory-Aware Transformer for Video Super-Resolution},
booktitle = {CVPR},
year = {2022},
month = {June}
}
``` -->

## Contact
If you meet any problems, please describe them in issues or contact:
* Chengxu Liu: <liuchx97@gmail.com>

