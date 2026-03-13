"""
Feature Map Heatmap Visualization

Generates heatmap visualizations of intermediate feature maps from the network,
enabling visual analysis of what different layers have learned.
Uses forward hooks to capture intermediate outputs and applies colormap overlays.
"""
import os
import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
import numpy as np
import sys
sys.path.append('../')
import copy
import matplotlib.cm as mpl_color_map
from lib import Net                       # Your network module
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # CUDA device
torch.manual_seed(2021)
to_pil = transforms.ToPILImage()
results = OrderedDict()


def check_mkdir(dir_name):
    """Create directory if it does not exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_cam(input_image, target):
    """Generate a class activation map (CAM) from feature maps.

    Args:
        input_image: Original PIL image (used for resizing).
        target: Feature map tensor to visualize.

    Returns:
        Normalized activation map as a numpy array.
    """
    cam = target.cuda().data.sum(axis=0).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = np.uint8(cam * 255)
    cam = np.uint8(Image.fromarray(cam[0]).resize(input_image.size, Image.ANTIALIAS)) / 255
    return cam


def apply_colormap_on_image(org_im, activation):
    """Apply a jet colormap on the activation map and overlay it on the original image.

    Args:
        org_im: Original PIL image.
        activation: Normalized activation map (numpy array).

    Returns:
        tuple: (no_trans_heatmap, heatmap_on_image) as PIL images.
    """
    color_map = mpl_color_map.get_cmap('jet')
    no_trans_heatmap = color_map(activation)

    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def hook_func(module, input, output, medium_results_path='./medium_results'):
    """Forward hook function that captures and saves feature map visualizations.

    Args:
        module: The hooked module.
        input: Module input.
        output: Module output (feature maps to visualize).
        medium_results_path: Directory to save visualization results.
    """
    global image_name, ori_img, cnt, module_name
    base_name = str(module).split('(')[0]
    check_mkdir(os.path.join(medium_results_path, base_name))
    check_mkdir(os.path.join(medium_results_path, base_name, module_name))
    save_name = os.path.join(medium_results_path, base_name, module_name, image_name + '.png')
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)
    data = get_cam(ori_img, data.cpu())
    no_ori_img, save_img = apply_colormap_on_image(ori_img, data)
    save_img.save(save_name)


def main(exp_name, net, pretrained_model_path, modules_for_plot, scale):
    """Run inference and generate feature map visualizations.

    Args:
        exp_name: Experiment name.
        net: Network model instance.
        pretrained_model_path: Path to the pretrained model checkpoint.
        modules_for_plot: OrderedDict mapping module names to module objects for hooking.
        scale: Input image resize scale.
    """
    to_test = OrderedDict([
        ('COD10K', './data/TestDataset/COD10K/'),
    ])  # Dataset name and path
    global cnt
    cnt = 0
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(pretrained_model_path))
    print('Load {} succeed!'.format(pretrained_model_path + '.pth'))
    net.eval()
    img_transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for name, module in modules_for_plot.items():
        global module_name
        module_name = name
        module.register_forward_hook(hook_func)

    with torch.no_grad():
        for name, root in to_test.items():
            image_path = os.path.join(root, 'Imgs')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in enumerate(img_list):
                global image_name, ori_img
                image_name = img_name
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')
                ori_img = img
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                predictions = net(img_var)
            print('{}'.format(exp_name))


def predict_model(modelname, pretrained_model_path, net):
    """Set up modules for visualization and run the main pipeline.

    Args:
        modelname: Model name for identification.
        pretrained_model_path: Path to pretrained weights.
        net: Network model instance.
    """
    modules_for_plot = OrderedDict([
        ('net.cam5.conv3_3', net.cam5.conv3_3),
    ])  # Network layers to visualize

    main(modelname, net, pretrained_model_path, modules_for_plot, scale=384)


if __name__ == '__main__':
    predict_model('NET', './checkpoints/PVT.pth', Net())  # Checkpoint path