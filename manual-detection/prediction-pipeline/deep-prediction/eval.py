import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from dice_loss import dice_coeff
import os
import numpy as np 

# so i can run on my mac; can be removed. 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def eval_net(net, loader, device, pathFile = None, saveImage = False):
    """Evaluation without the densecrf with the dice coefficient"""

    net.eval()

    mask_type = torch.float32

    n_val = len(loader)   
    tot = 0

   
    for batch in loader:
        imgs, true_masks, filename = batch['image'], batch['mask'], batch['filename']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        mask_pred = mask_pred['out']
        pred = mask_pred.max(dim=1)[1]
        pred = (pred).float()
        tot += dice_coeff(pred, true_masks).item()
      
      
        if saveImage:  

          
            output_seg = mask_pred.max(dim=1)[1].unsqueeze(1)
            output_seg = output_seg.data.cpu().numpy() 


            if not os.path.exists(pathFile):
               
                os.makedirs(pathFile)

            for i in range(output_seg.shape[0]):
        
                output_img = output_seg[i,0,:,:] * 255
               
                filePath = os.path.join(pathFile, filename[i] + ".png")
                Image.fromarray(output_img.astype(np.uint8)).save(filePath)
                
    return tot / n_val
