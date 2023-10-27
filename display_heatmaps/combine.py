import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.sff import net      ##import net
from PIL import Image
import torchvision.transforms as transforms

import logging
import imageio

parser = argparse.ArgumentParser()

parser.add_argument('--pth_path', type=str, default='*************')  #checkpoints-path
opt = parser.parse_args()

class My_combine_dataset:
    def __init__(self, image_root, gt_root):

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
           
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

def combine_mask_and_image(mask1,img): #combine

    img = np.asarray(img)                     
    mask1 = np.asarray(mask1)                    
    mask1 = cv2.resize(mask1,(img.shape[1],img.shape[0]))  #
    mask = np.zeros(img.shape)   
    ####################################################### RGB             
    mask[:, :, 0] = np.where(mask1 > 235, 255, 0)
    mask[:, :, 1] = 0
    mask[:, :, 2] = 0 #np.where(mask1 > 200, 255, 0)
    ####################################################### RGB
    alpha = 0.3
    beta = 1 - alpha
    gamma = 0
    new_image = cv2.addWeighted(img, alpha, mask, beta, gamma,dtype=cv2.CV_8U)
   # mask1 = mask1.expand(mask1.shape[0],mask1.shape[1],3)
    new_image = np.where(mask>235,new_image,img)
    return new_image

# for _data_name in ['CAMO', 'COD10K', 'CHAMELEON']:
for _data_name in ['CAMO', 'COD10K', 'NC4K']:                 ##dataname
    data_path = './data/{}/'.format(_data_name)     ##data path
    save_path = './data//{}/'.format(_data_name)    ##save path
    image_root = '{}Imgs/'.format(data_path)        ##image_root
    mask_path = './data//{}/'.format(_data_name)    ##mask_path
    gt_root = '{}GT/'.format(data_path)

    combine_loader = My_combine_dataset(image_root, gt_root)
    print('****',combine_loader.size)
    for i in range(combine_loader.size):
        image, gt, name = combine_loader.load_data()
        # print('***name',name)
        new_img = combine_mask_and_image(gt, image)
        # print('***save_path',save_path)
        imageio.imwrite(save_path+name, (new_img).astype(np.uint8))
        
print('root',image_root,gt_root,mask_path)