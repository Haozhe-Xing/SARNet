"""
Prediction Map Combination Visualization

Overlays model prediction masks on original images with colored highlighting,
useful for visualizing camouflaged object detection results.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.sff import net      # Import network
from PIL import Image
import torchvision.transforms as transforms

import logging
import imageio

parser = argparse.ArgumentParser()

parser.add_argument('--pth_path', type=str, default='*************')  # Checkpoint path
opt = parser.parse_args()


class My_combine_dataset:
    """Dataset class for loading image-GT pairs for visualization."""

    def __init__(self, image_root, gt_root):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        """Load the next image-GT pair."""
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        """Load an RGB image."""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        """Load a grayscale binary mask."""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


def combine_mask_and_image(mask1, img):
    """Overlay a prediction mask on the original image with red highlighting.

    Args:
        mask1: Prediction mask (PIL Image or numpy array).
        img: Original image (PIL Image or numpy array).

    Returns:
        Combined image as a numpy array.
    """
    img = np.asarray(img)
    mask1 = np.asarray(mask1)
    mask1 = cv2.resize(mask1, (img.shape[1], img.shape[0]))
    mask = np.zeros(img.shape)
    # Red channel highlighting for detected regions
    mask[:, :, 0] = np.where(mask1 > 235, 255, 0)
    mask[:, :, 1] = 0
    mask[:, :, 2] = 0
    alpha = 0.3
    beta = 1 - alpha
    gamma = 0
    new_image = cv2.addWeighted(img, alpha, mask, beta, gamma, dtype=cv2.CV_8U)
    new_image = np.where(mask > 235, new_image, img)
    return new_image


# Dataset names to process
for _data_name in ['CAMO', 'COD10K', 'NC4K']:
    data_path = './data/{}/'.format(_data_name)       # Data path
    save_path = './data//{}/'.format(_data_name)      # Save path
    image_root = '{}Imgs/'.format(data_path)          # Image root
    mask_path = './data//{}/'.format(_data_name)      # Mask path
    gt_root = '{}GT/'.format(data_path)

    combine_loader = My_combine_dataset(image_root, gt_root)
    print('****', combine_loader.size)
    for i in range(combine_loader.size):
        image, gt, name = combine_loader.load_data()
        new_img = combine_mask_and_image(gt, image)
        imageio.imwrite(save_path + name, (new_img).astype(np.uint8))

print('root', image_root, gt_root, mask_path)