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

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:  # not valid grammar in Python 2.5
        yaml_cfg = edict(yaml.load(f), Loader=yaml.FullLoader)
    return yaml_cfg


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    ind_p = target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0
    log_p = log_p[ind_p.view(-1,c )]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input, dim=1)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)  # b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[:, 1]  # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total

def convert_result_to_csv(values, csv_file_name):
    if not os.path.exists(csv_file_name):
        df = DataFrame(columns=['epoch', 'val_dice', 'test_dice', 'external_test_dice'])
        df_i = 0
    else:
        df = pd.read_csv(csv_file_name)
        df_i = len(df)
    df.loc[df_i] = [values[0], values[1], values[2], values[3]]
    df.to_csv(csv_file_name, index=False)


def train_net(net,
              device,
              epochs=250,
              batch_size=4,
              lr=0.0001,
              save_cp=True,
              args=None,
              input_path=None):

    #assign image path
    dir_img = input_path.dir_img
    dir_mask = input_path.dir_mask
    dir_valimg = input_path.dir_valimg
    dir_valmask = input_path.dir_valmask
    dir_testimg = input_path.dir_testimg
    dir_testmask = input_path.dir_testmask
    dir_externaltestimg = input_path.dir_externaltestimg
    dir_externaltestmask = input_path.dir_externaltestmask

    #assign experimental options
    exp_name = args.expname
    img_scale = args.scale
    color_map = args.colormap

    dir_checkpoint = os.path.join(input_path.dir_checkpoint, exp_name)

    dataset = BasicDataset(dir_img, dir_mask, img_scale,color_map,'train')
    dataval = BasicDataset(dir_valimg, dir_valmask, img_scale,color_map,'val')
    datatest = BasicDataset(dir_testimg, dir_testmask, img_scale,color_map, 'test')
    dataexternaltest = BasicDataset(dir_externaltestimg, dir_externaltestmask, img_scale,color_map, 'test')

    n_val = dataval.__len__()
    n_train = dataset.__len__()

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last = False)
    val_loader = DataLoader(dataval, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(datatest, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    external_test_loader = DataLoader(dataexternaltest, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    writer = SummaryWriter(comment=f'_EXPNAME_{exp_name}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Color map:       {color_map}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
 
    for epoch in range(epochs):

        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
      
                true_masks_2channel = true_masks.unsqueeze(1)
                true_masks_2channel = torch.cat((1-true_masks_2channel, true_masks_2channel), 1)
                true_masks_2channel = true_masks_2channel.to(device=device, dtype=torch.float32)
        
                assert imgs.shape[1] == 3, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # need to convert the true masks into a binary array, and not an ordered dictionary. 
                masks_pred = net(imgs)
                masks_pred = masks_pred['out']
                
                loss_cross = criterion(masks_pred, true_masks)
                loss_dice = 1 + dice_loss(masks_pred, true_masks_2channel)
                loss = loss_dice + loss_cross

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_cr (batch)': loss_cross.item(), 'loss_dsc (batch)': loss_dice.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])

        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

        val_score = eval_net(net, val_loader, device)
        test_score = eval_net(net, test_loader, device)
        external_test_score = eval_net(net, external_test_loader, device)

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        logging.info('Finish Epoch %d/%d'% (epoch,epochs))
        logging.info('Validation Dice Coeff: {}'.format(val_score))
        writer.add_scalar('Dice/test', val_score, global_step)
        logging.info('Internal Testing Dice Coeff: {}'.format(test_score))
        writer.add_scalar('Loss/test', test_score, global_step)
        logging.info('External Testing Dice Coeff: {}'.format(external_test_score))
        writer.add_scalar('Loss/test', external_test_score, global_step)

        writer.add_images('images', imgs, global_step)

        writer.add_images('masks/true', true_masks.unsqueeze(1), global_step)
        writer.add_images('masks/pred', masks_pred.max(dim=1)[1].unsqueeze(1), global_step)
    
        if not os.path.exists(dir_checkpoint):
            os.makedirs(dir_checkpoint)

        csv_file_name = os.path.join(dir_checkpoint, '%s_result_log.csv' % exp_name)
        convert_result_to_csv([epoch, val_score, test_score, external_test_score], csv_file_name)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='UNet_exp0.yml', help='config file')


    opt = parser.parse_args()
    config_file = os.path.join('config', opt.config)

    cfg = cfg_from_file(config_file)

    input_path = edict()
    input_path.dir_img = cfg.INPUT.DIR_IMG
    input_path.dir_mask = cfg.INPUT.DIR_MASK
    input_path.dir_valimg = cfg.INPUT.DIR_VALIMG
    input_path.dir_valmask = cfg.INPUT.DIR_VALMASK
    input_path.dir_testimg = cfg.INPUT.DIR_TESTIMG
    input_path.dir_testmask = cfg.INPUT.DIR_TESTMASK
    input_path.dir_externaltestimg = cfg.INPUT.DIR_EXTTESTIMG
    input_path.dir_externaltestmask = cfg.INPUT.DIR_EXTTESTMASK
    input_path.dir_checkpoint = cfg.INPUT.DIR_CHECKPOINT

    args = edict()
    args.epochs = cfg.ARGS.EPOCHS
    args.batchsize = cfg.ARGS.BATCH_SIZE
    args.lr = cfg.ARGS.LR
    args.load_pth = cfg.ARGS.LOAD_PTH

    args.expname = opt.config.replace('.yml','')
    args.scale = cfg.EXP.SCALE
    args.colormap = cfg.EXP.COLORMAP


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=False, num_classes = 2)

    if args.load_pth:
        net.load_state_dict(
            torch.load(args.load_pth, map_location=device)
        )
        logging.info(f'Model loaded from {args.load_pth}')

    net.to(device=device)
  
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  args =args,
                  input_path=input_path)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)