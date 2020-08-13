import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
#yuankai change to tensorboard
from tensorboardX import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from pandas import DataFrame
import pandas as pd
import csv 

if __name__ == '__main__':


    results = []

    dir_img = 'masks_test/'
    dir_mask = 'Output/'

    # the img scale arg is irrelevant here. 
    img_scale = 4
    color_map = 'RGB'
    # batch size must be 1 as the images are not of the same size. 
    batch_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BasicDataset(dir_img, dir_mask, img_scale,color_map,'train')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    dice = eval_net(train_loader, device, results, color_map)

    fields = ['Dice']

    rows = []
    for x in results: 
      rows.append(x)


    filename = "UNET_Output.csv"

    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        
        # writing the fields 
        csvwriter.writerow(fields) 
        
        # writing the data rows 
        csvwriter.writerows(rows)

    print("Completed")