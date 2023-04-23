#python -m tensorboard.main --logdir VPTR_ckpts/MNIST_ResNetAE_MSEGDLgan_tensorboard
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import torch.nn.functional as F

import tifffile

from pathlib import Path
import random
from datetime import datetime
from time import sleep
from tqdm import tqdm

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights
from model import GDL, MSELoss, L1Loss, GANLoss
from utils import get_dataloader, get_data
from utils import VidCenterCrop, VidPad, VidResize, VidNormalize, VidReNormalize, VidCrop, VidRandomHorizontalFlip, VidRandomVerticalFlip, VidToTensor
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, update_summary, write_summary, resume_training
from utils import set_seed
import os

from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__':
    run = "2"
    working_dir = '/gpfs/home/shared/Neurotic/'
    ckpt_save_dir = Path(working_dir+'trained_ae_'+run)

    # resume_ckpt = ckpt_save_dir.joinpath('epoch_2.tar')
    resume_ckpt = None
    start_epoch = 0

    num_past_frames = 75
    num_future_frames = 25
    encH, encW, encC = 8, 8, 528
    img_channels = 1 #3 channels for BAIR datset
    epochs = 50
    N = 1
    AE_lr = 2e-4
    lam_gan = 0.01
    device = torch.device('cpu')

    #####################Init Dataset ###########################
    data_set_name = 'CSD' #see utils.dataset
    dataset_dir = working_dir+'MCS'
    train_loader, val_loader, test_loader, renorm_transform = get_data(N, dataset_dir, num_frames = 20, video_limit = 2000)

    #####################Init Models and Optimizer ###########################
    VPTR_Enc = VPTREnc(img_channels, feat_dim = encC, n_downsampling = 3).to(device)
    VPTR_Dec = VPTRDec(img_channels, feat_dim = encC, n_downsampling = 3, out_layer = 'Sigmoid').to(device) #Sigmoid for MNIST, Tanh for KTH and BAIR
    VPTR_Disc = VPTRDisc(img_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
    init_weights(VPTR_Disc)
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)

    optimizer_G = torch.optim.Adam(params = list(VPTR_Enc.parameters()) + list(VPTR_Dec.parameters()), lr=AE_lr, betas = (0.5, 0.999))
    optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr=AE_lr, betas = (0.5, 0.999))

    Enc_parameters = sum(p.numel() for p in VPTR_Enc.parameters() if p.requires_grad)
    Dec_parameters = sum(p.numel() for p in VPTR_Dec.parameters() if p.requires_grad)
    Disc_parameters = sum(p.numel() for p in VPTR_Disc.parameters() if p.requires_grad)
    print(f"Encoder num_parameters: {Enc_parameters}")
    print(f"Decoder num_parameters: {Dec_parameters}")
    print(f"Discriminator num_parameters: {Disc_parameters}")

    #####################Init Criterion ###########################
    loss_name_list = ['AE_MSE', 'AE_GDL', 'AE_total', 'Dtotal', 'Dfake', 'Dreal', 'AEgan']
    gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device)
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss()
    gdl_loss = GDL(alpha = 1)

    VPTR_Dec = None
    VPTR_Enc = None
    VPTR_Disc = None
    #####################Training loop ###########################                                            
    with tqdm(enumerate(train_loader, 0), unit=" batch", total=len(train_loader)) as tepoch:
        frames = []
        subtitles = []
        for idx, sample in tepoch:
            pf, ff, d = sample
            print("Maximum value:", torch.max(ff))
            print("Minimum value:", torch.min(ff))
            print("Mean value:", torch.mean(ff))
            print()

            ff = renorm_transform(ff[0])
            first_frame = np.array(ff[-1,0,:])
            
            if np.all(first_frame == 0) or np.all(first_frame == 1) or np.all(np.isnan(first_frame)):
                print("Array is blank")
                print(d)

            # Convert the pixel data to grayscale
            gray_frame = (first_frame * 255).astype('uint8')
            # Resize the image to 100x100 pixels
            pil_image = Image.fromarray(gray_frame)
            resized_image = pil_image.resize((100, 100))

            # Convert the resized image back to a numpy array
            resized_frame = np.array(resized_image)

            frames.append(resized_frame)
            subtitles.append(str(d))

        # Create a grid of images with titles
        num_cols = 20
        num_rows = int(np.ceil(len(frames) / num_cols))
        img_size = 100
        grid_img = Image.new(mode='L', size=(num_cols*img_size, num_rows*img_size), color='white')

        for i, frame in enumerate(frames):
            row = i // num_cols
            col = i % num_cols
            img = Image.fromarray(frame, mode='L')
            img = img.resize((img_size, img_size), resample=Image.NEAREST)
            grid_img.paste(img, (col*img_size, row*img_size))

        # Add subtitles
        draw = ImageDraw.Draw(grid_img)
        font_size = 24
        font = ImageFont.truetype("/home/shared/Neurotic/vptr/VFFP/arial.ttf", size=font_size)
        for i, subtitle in enumerate(subtitles):
            row = i // num_cols
            col = i % num_cols
            draw.text((col*img_size, row*img_size + img_size - font_size - 5), subtitle, font=font, fill='black')

    # Save the grid image as a PNG file

    save_dir = os.path.join(os.getcwd(), "dataset_test")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = save_dir + '/all_grid.png'
    grid_img.save(save_path)