import torch
import torch.nn.functional as F
from tqdm import tqdm
from dice_loss import dice_coeff
from dice_loss import DiceCoeff

import os 

def eval_net(loader, device, dict, color_map):

    mask_type = torch.float32
    n_val = len(loader)   
    tot = 0

    for batch in loader:
        imgs, true_masks, name = batch['image'], batch['mask'], batch['name']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        tot += dice_coeff(imgs, true_masks).item()
        
        for x in range(len(imgs)):
   
            test = DiceCoeff().forward(imgs[x], true_masks[x]).item()
            dict.append([test])

    print("total:: " + str(tot/n_val))
    return tot / n_val