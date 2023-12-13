# print("Importing Libraries")
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.commons import Weights_Normal, VGG19_PercepLoss
from models.MuLA_GAN import MuLA_GAN_Generator, Discriminator
from utils.data_utils import Dataloader

# print("Importing Done")

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_MuLA-GAN.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=301, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
args = parser.parse_args()

# print("Reading done")

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2 
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"] 
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"] 
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]

# print("##############")
# create dir for model and validation data
samples_dir = os.path.join("samples/", dataset_name)
checkpoint_dir = os.path.join("checkpoints/", dataset_name)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

Adv_cGAN = torch.nn.MSELoss()
L1_G  = torch.nn.L1Loss() 
L_vgg = VGG19_PercepLoss() 
lambda_1, lambda_con = 7, 3 
patch = (1, img_height//16, img_width//16)
# print("111@@@@@@@@@@@@@@")
# Initialize generator and discriminator
generator = MuLA_GAN_Generator()
discriminator = Discriminator()
# print("222@@@@@@@@@@@@@@")
# see if cuda is available
if torch.cuda.is_available():
    # print("333@@@@@@@@@@@@@@")
    generator = generator.cuda()
    # print("444@@@@@@@@@@@@@@")
    discriminator = discriminator.cuda()
    Adv_cGAN.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    Tensor = torch.cuda.FloatTensor
    
else:
    Tensor = torch.FloatTensor

print("@@@@@@@@@@@@@@")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))

## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


dataloader = DataLoader(
    Dataloader(dataset_path, dataset_name, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
)


## Training pipeline
for epoch in range(epoch, num_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

        ## Train Discriminator
        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt, imgs_distorted)
        loss_real = Adv_cGAN(pred_real, valid)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_fake = Adv_cGAN(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake) * 10.0 
        loss_D.backward()
        optimizer_D.step()

        ## Train Generator
        optimizer_G.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_GAN =  Adv_cGAN(pred_fake, valid) 
        loss_1 = L1_G(imgs_fake, imgs_good_gt)
        loss_con = L_vgg(imgs_fake, imgs_good_gt)
        loss_G = loss_GAN + lambda_1 * loss_1  + lambda_con * loss_con 
        loss_G.backward()
        optimizer_G.step()

        ## Print log
        if not i%50:
            sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f]"
                              %(
                                epoch, num_epochs, i, len(dataloader),
                                loss_D.item(), loss_G.item(), loss_GAN.item(),
                               )
            )
        batches_done = epoch * len(dataloader) + i
    ## Save model checkpoints
    if (epoch % ckpt_interval == 0):
        torch.save(generator.state_dict(), "checkpoints/%s/generator_%d.pth" % (dataset_name, epoch))
        torch.save(discriminator.state_dict(), "checkpoints/%s/discriminator_%d.pth" % (dataset_name, epoch))


