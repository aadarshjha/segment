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
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        random.seed(1)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def augmentation(self, pil_img, pil_mask):
        input_img = np.expand_dims(pil_img, axis=0)
        input_mask = np.expand_dims(pil_mask, axis=0)
        input_mask = np.expand_dims(input_mask, axis=3)

        prob = random.uniform(0, 1)
        if self.split == 'train' and prob > 0.5:# we do augmentation in 50% of the cases
            seq = iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace=self.color_map),
                iaa.ChannelShuffle(0.35),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                iaa.Affine(rotate=(-180, 180)),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.GaussianBlur(sigma=(0, 3.0))
            ])
            images_aug, segmaps_aug = seq(images=input_img, segmentation_maps=input_mask)

            output_img = np.transpose(images_aug[0], (2, 0, 1))
            output_mask = np.transpose(segmaps_aug[0], (2, 0, 1))
        else:
            seq = iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace=self.color_map),
            ])
            images_aug, segmaps_aug = seq(images=input_img, segmentation_maps=input_mask)
            output_img = np.transpose(images_aug[0], (2, 0, 1))
            output_mask = np.transpose(segmaps_aug[0], (2, 0, 1))



        return output_img, output_mask

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0]).convert('L')
        img = Image.open(img_file[0])

        if self.scale < 1:
            w, h = img.size
            newW, newH = int(self.scale * w), int(self.scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            img = img.resize((newW, newH))
            mask = mask.resize((newW, newH))

        mask = np.array(mask)
        mask[mask>0] = 1
        mask = Image.fromarray(mask)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = np.array(img)
        mask = np.array(mask)
        [img,mask] = self.augmentation(img, mask)

        if img.max() > 1:
            img = img / 255

        mask = mask[0,:]

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}