# modified FOR 512 against something comparison. 
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff
from dice_loss import DiceCoeff

import numpy as np
from scipy.ndimage import morphology
from PIL import Image

import os 

# so i can run on my mac; can be removed. 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def eval_net(loader, device, dict, color_map):


    # simple dice script for computing the values; 
    # dataloader has also been modified; 
    mask_type = torch.float32
    n_val = len(loader)   
    tot = 0

    for batch in loader:
        imgs, true_masks, name = batch['image'], batch['mask'], batch['name']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)


        # getting each image. 
        tot += dice_coeff(imgs, true_masks).item()
        
        for x in range(len(imgs)):
           
            # calculating individual dice scores: 
            # test
    
            test = DiceCoeff().forward(imgs[x], true_masks[x]).item()
            # yeet = dice_coeff(imgs[x], true_masks[x]).item()
            # print(test)

            # checking if not in keys: 
            dict.append([test])


    print("total:: " + str(tot/n_val))
    return tot / n_val


