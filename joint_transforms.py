import random

from PIL import Image
from PIL.Image import frombytes
from torchvision import transforms
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Compose3(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask,edge):
        assert img.size == mask.size, img.size == edge.size
        for t in self.transforms:
            img, mask, edge = t(img, mask,edge)
        return img, mask, edge

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)

        # elif random.random() < 0.6:
        #     return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)

        else:
            return img, mask

class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

class RandomHorizontallyFlip3(object):
    def __call__(self, img, mask, edge):

        if random.random() < 0.3:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT),edge.transpose(Image.FLIP_LEFT_RIGHT)
        elif random.random() < 0.6:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM), edge.transpose(
                Image.FLIP_TOP_BOTTOM)
        else:
            return img, mask, edge

class Resize3(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask, edge):
        assert img.size == mask.size and img.size == edge.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), edge.resize(self.size, Image.BILINEAR)



class RandomResizedCrop_transpose(object):
    """
    Randomly crop and resize the given image with a probability of 0.5
    """
    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between 0 and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.size
            #print(img.size)
            x1 = random.randint(0, max(0, w - self.ch))
            y1 = random.randint(0, max(0, h - self.cw))
            img_crop = img.crop((y1, x1, y1 + self.cw, self.ch + x1))
            label_crop = label.crop((y1, x1, y1 + self.cw, self.ch + x1))

            #img_crop = img_crop.resize((w, h))
            #label_crop = label_crop.resize((w, h))
            return img_crop, label_crop
        else:
            return img, label
