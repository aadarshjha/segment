import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()

    mask_type = torch.float32

    n_val = len(loader) 
    tot = 0

    for batch in loader:
      
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        mask_pred = mask_pred['out']
        pred = mask_pred.max(dim=1)[1]
        pred = (pred).float()
        tot += dice_coeff(pred, true_masks).item()
    
    return tot / n_val
