"""
Dataset Loading Module

Provides dataset classes for Camouflaged Object Detection (COD) tasks,
supporting both training and testing modes.

Expected directory structure:
    root/
    ├── Imgs/       # Input images (.jpg or .png)
    └── GT/         # Ground truth masks (.png, same name as images)
"""

import os
import os.path
import torch.utils.data as data
from PIL import Image


def make_dataset(root):
    """Build a list of image-mask pairs.

    Supports both a single dataset directory or a list of multiple directories.

    Args:
        root: Dataset root directory path (str) or a list of directory paths (list).

    Returns:
        list: A list of (image_path, mask_path) tuples.
    """
    if isinstance(root, str):
        # Single dataset directory
        image_path = os.path.join(root, 'Imgs')
        mask_path = os.path.join(root, 'GT')
        img_list = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('png')]
        imgs = [
            (os.path.join(image_path, img_name),
             os.path.join(mask_path, os.path.splitext(img_name)[0] + '.png'))
            for img_name in img_list
        ]
        return imgs
    else:
        # Multiple dataset directories: merge all image-mask pairs
        imgs = []
        for root_dir in root:
            image_path = os.path.join(root_dir, 'Imgs')
            mask_path = os.path.join(root_dir, 'GT')
            img_list = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('png')]
            sub_imgs = [
                (os.path.join(image_path, img_name),
                 os.path.join(mask_path, os.path.splitext(img_name)[0] + '.png'))
                for img_name in img_list
            ]
            imgs.extend(sub_imgs)
        return imgs


class ImageFolder(data.Dataset):
    """Training dataset class.

    Loads images and corresponding ground truth masks, supporting joint transforms
    (e.g., synchronized flipping, cropping), image transforms, and target transforms.

    Args:
        root: Dataset root directory path or a list of paths.
        joint_transform: Transforms applied synchronously to both image and mask
            (e.g., random flip, resize).
        transform: Transforms applied only to the image (e.g., color jitter, normalization).
        target_transform: Transforms applied only to the mask (e.g., ToTensor).
    """

    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """Retrieve the sample at the given index.

        Returns:
            tuple: (image_tensor, mask_tensor)
        """
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_test(data.Dataset):
    """Testing dataset class.

    Loads images only (without masks), used during inference.

    Args:
        root: Dataset root directory path.
        transform: Transforms applied to the image.
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.transform = transform

    def __getitem__(self, index):
        """Retrieve the image at the given index.

        Returns:
            Transformed image tensor.
        """
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    print(isinstance('path', str))