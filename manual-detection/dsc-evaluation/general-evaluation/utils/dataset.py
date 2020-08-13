from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

import imgaug.augmenters as iaa
import random

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, color_map='RGB', split='train'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.split = split
        self.color_map = color_map
      
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        random.seed(1)

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, i):
        idx = self.ids[i]

        mask_file = glob(self.masks_dir + idx + '.png')
        img_file = glob(self.imgs_dir + idx + '.png')

        # skip over photo if we cant find a  pair 
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0]).convert('L')
       
        w, h = mask.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        mask = mask.resize((newW, newH))

  
        img = np.array(img)
        img[img>0] = 1
        img = Image.fromarray(img)


        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = np.array(img)
        mask = np.array(mask)

        if mask.max() > 1:
            mask = mask / 255
        
        return {'image': img, 'mask': mask, 'name': idx}