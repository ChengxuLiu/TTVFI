import random
import math
import torch.utils.data as data
import os
import numpy as np
import random
from random import choice
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Vimeo_90K_loader(Dataset):
    def __init__(self, data_root, is_training , input_frames="13"):##########
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training
        self.inputs = input_frames

        if self.training:
            # train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
            train_fn = os.path.join(self.data_root, 'tri_testlist2.txt')
            with open(train_fn, 'r') as f:
                self.trainlist = f.read().splitlines()
        else:
            test_fn = os.path.join(self.data_root, 'tri_testlist2.txt')
            with open(test_fn, 'r') as f:
                self.testlist = f.read().splitlines()

        if self.training:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        
        imgpaths = [imgpath + f'/im{i}.png' for i in range(1,4)]##########
        # Load images
        images = [Image.open(pth) for pth in imgpaths]
        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
            imagess = []
            for img_ in images:
                random.seed(seed)
                imagess.append(self.transforms(img_))
            images = imagess
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]
        else:
            imagess = []
            for img_ in images:
                imagess.append(self.transforms(img_))
            images = imagess
        return images[0],images[1],images[2]
        

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)

