"""
Joint Transforms Module

Provides synchronized image-mask transform operations for data augmentation
in segmentation tasks. Both image and mask undergo identical geometric
transformations to maintain spatial consistency.
"""
import random

from PIL import Image
from PIL.Image import frombytes
from torchvision import transforms


class Compose(object):
    """Compose multiple joint transforms for image-mask pairs."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Compose3(object):
    """Compose multiple joint transforms for image-mask-edge triplets."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, edge):
        assert img.size == mask.size, img.size == edge.size
        for t in self.transforms:
            img, mask, edge = t(img, mask, edge)
        return img, mask, edge


class RandomHorizontallyFlip(object):
    """Randomly flip image and mask horizontally with probability 0.5."""

    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img, mask


class Resize(object):
    """Resize image and mask to the specified size.

    Args:
        size: Target size as (h, w).
    """

    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w) -> PIL: (w, h)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class RandomHorizontallyFlip3(object):
    """Randomly flip image-mask-edge triplet horizontally or vertically."""

    def __call__(self, img, mask, edge):
        if random.random() < 0.3:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(Image.FLIP_LEFT_RIGHT)
        elif random.random() < 0.6:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM), edge.transpose(
                Image.FLIP_TOP_BOTTOM)
        else:
            return img, mask, edge


class Resize3(object):
    """Resize image-mask-edge triplet to the specified size.

    Args:
        size: Target size as (h, w).
    """

    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w) -> PIL: (w, h)

    def __call__(self, img, mask, edge):
        assert img.size == mask.size and img.size == edge.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), edge.resize(self.size, Image.BILINEAR)


class RandomResizedCrop_transpose(object):
    """Randomly crop and resize the given image-mask pair with probability 0.5.

    Args:
        crop_area: Maximum crop area size (used for both width and height).
    """

    def __init__(self, crop_area):
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.size
            x1 = random.randint(0, max(0, w - self.ch))
            y1 = random.randint(0, max(0, h - self.cw))
            img_crop = img.crop((y1, x1, y1 + self.cw, self.ch + x1))
            label_crop = label.crop((y1, x1, y1 + self.cw, self.ch + x1))
            return img_crop, label_crop
        else:
            return img, label