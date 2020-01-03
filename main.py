import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import time

from architectures import get_architecture
from my_loader import MyCustomDataset

parser = argparse.ArgumentParser(description='PyTorch Ensemble Attack')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size for attack (default: 1)')
parser.add_argument('--epsilon', default = 0.125,type = float, 
                    help='perturbation, (default: 0.125)')
parser.add_argument('--num_steps', default=40,type=int,
                    help='perturb number of steps, (default: 20)')
parser.add_argument('--step_size', default = 0.031, type=float,  help='perturb size')
parser.add_argument('--beta', default = 5.0, type=float, help='trade-off between target and non-target loss, (default: 5)')
parser.add_argument('--img_path', default = "./../../images/", type=str, help='path of the images')
parser.add_argument('--csv_path', default = "dev.csv", type=str, help='path of the csv')
parser.add_argument('--random', default = 1, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Average_logits(model_list, img):
    out = torch.zeros(len(model_list), 1000).cuda()
    item = 0
    for model in model_list:
        out[item, :] = model(img)
        item += 1
    return torch.mean(out, dim = 0, keepdim = True)

def PGD_ms_attack(model_list, x_nature, y, target, step_size, epsilon, perturb_steps, beta, img_name, random):
    if random:
        random_noise = torch.FloatTensor(*x_nature.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(x_nature.data + random_noise, requires_grad=True)
    else:
        X_pgd = Variable(x_nature.data, requires_grad=True)

    decay_1 = int(perturb_steps / 2)
    decay_2 = int(perturb_steps * 3 / 4)  
    lr = step_size
    
    for step in range(perturb_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            h, w = X_pgd.shape[-2:]
            out = Average_logits(model_list, X_pgd)
            loss = F.cross_entropy(out, y)  - beta * F.cross_entropy(out, target)
            for idx, scale in enumerate((0.74, 1.25)):
                # resizes
                size = tuple([int(s * scale) for s in (h, w)])
                x_resize = F.interpolate(X_pgd, size=size, mode="bilinear", align_corners=True)
                out = Average_logits(model_list, x_resize)
                loss += F.cross_entropy(out, y)  - beta * F.cross_entropy(out, target)
        # print("Step: {}, Loss: {}".format(step, loss.data))
        loss.backward()
        eta = lr * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - x_nature.data, -epsilon, epsilon)
        X_pgd = Variable(x_nature.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=True)

        if (step + 1) >= decay_1:
            lr = 0.5 * step_size
            decay_1 = perturb_steps + 1
        
        if (step + 1) >= decay_2:
            lr = 0.25 * step_size
            decay_2 = perturb_steps + 1

    return X_pgd.data

if __name__ == "__main__":

    resnet152 = get_architecture(denoise=False).cuda()
    resnet152.eval()

    resnet152_denoise = get_architecture(denoise=True).cuda()
    resnet152_denoise.eval()

    resnet101_denoise = get_architecture(denoise=True, model_name="Resnet101-DenoiseAll").cuda()
    resnet101_denoise.eval()

    model_list = [resnet152, resnet152_denoise, resnet101_denoise]

    loader = MyCustomDataset(csv_path=args.csv_path, img_path=args.img_path)
    
    attack_loader = torch.utils.data.DataLoader(dataset=loader, 
                                                batch_size=args.batch_size, 
                                                shuffle=False, 
                                                sampler=torch.utils.data.SequentialSampler(loader))
    
    record = True
    for (img, label, target, img_name) in attack_loader:
        img, label, target = img.to(device), label.to(device), target.to(device)
        
        save_dir = "images_main_attack/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if os.path.exists(save_dir + img_name[0]):
            continue
        
        if record:
            start = time.clock()

        x_adv = PGD_ms_attack(model_list=model_list,
                              x_nature=img,
                              y=label,
                              target=target,
                              step_size=args.step_size,
                              epsilon=args.epsilon,
                              perturb_steps=args.num_steps,
                              beta=args.beta, 
                              img_name=img_name[0],
                              random=args.random)
        
        if record:
            end = time.clock()
            print(end-start)
            record = False

        img_adv = transforms.ToPILImage()(x_adv[0, :, :, :].cpu()).convert('RGB')
        img_adv.save(os.path.join(save_dir, img_name[0]))