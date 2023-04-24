import os
import os.path
import torch.utils.data as data
from PIL import Image

def make_dataset(root):
    if isinstance(root,str):
        image_path = os.path.join(root, 'Imgs')
        #image_path = os.path.join(root, 'without_Edge')
        mask_path = os.path.join(root, 'GT')
        img_list = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('png')]
        imgs = [(os.path.join(image_path, img_name), os.path.join(mask_path, os.path.splitext(img_name)[0] + '.png')) for img_name in img_list]
        return imgs
    else:
        imgs = []
        for root_dir in root:
            image_path = os.path.join(root_dir, 'Imgs')
            # image_path = os.path.join(root, 'without_Edge')
            mask_path = os.path.join(root_dir, 'GT')
            img_list = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('png')]
            sub_imgs = [
                (os.path.join(image_path, img_name), os.path.join(mask_path, os.path.splitext(img_name)[0] + '.png'))
                for img_name in img_list]
            imgs.extend(sub_imgs)
        return imgs

class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.imgs = self.imgs
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
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
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.transform = transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    print(isinstance('path',str))