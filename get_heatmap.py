import os
import time
import datetime

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
import sys
sys.path.append('../')
from config import *
from misc import *
torch.manual_seed(2021)
print(torch.__version__)

to_pil = transforms.ToPILImage()
to_test = OrderedDict([
    ('CHAMELEON', chameleon_path),
    ('CAMO', camo_path),
    ('COD10K', cod10k_path),
    ('NC4K', nc4k_path)
])

results = OrderedDict()
image_name = 0
import cv2
import math
import copy
import matplotlib.cm as mpl_color_map
medium_results_path ='./analyze'

def get_cam(input_image,target):
    # Multiply each weight with its conv output and then, sum
    # for i in range(len(target)):
    #     cam += target[i, :, :].data.numpy()

    #利用cuda加速
    cam = target.cuda().data.sum(axis = 0).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    cam = np.uint8(Image.fromarray(cam[0]).resize(input_image.size, Image.ANTIALIAS)) / 255
    return cam


def apply_colormap_on_image(org_im, activation):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap('jet')
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def get_activation(name):
    def hook_func(module, input, output):
        """
        Hook function of register_forward_hook

        Parameters:
        -----------
        module: module of neural network
        input: input of module
        output: output of module
        """
        # save_image()每一行的特征图数量默认是8，可以通过nrow参数进行修改。
        # pad_value是特征图中间填充的间隙的颜色，尽管PyTorch将其定义为整数，但是由于Tensor一般是float型，所以还是输入个[0, 1]之间的浮点数才能真正生效。
        #image_name = get_image_name_for_hook(module)
        global image_name, cnt, ori_img, ori_img_path

        base_name = str(module).split('(')[0]
        check_mkdir(os.path.join(medium_results_path,base_name))
        img = cv2.imread(ori_img_path)
        if base_name =='Transformer1':
            save_dir = os.path.join(medium_results_path, base_name+'_'+str(module.dim))
            check_mkdir(save_dir)
            save_name = os.path.join(save_dir,image_name + '.png')
            data = output.clone().detach()
            n, hw, c = data.shape
            h = w = int(math.sqrt(hw))
            data = data.transpose(1, 2).reshape(n, c, h, w)
            data = data.permute(1, 0, 2, 3)
            data = get_cam(ori_img, data.cpu())
            no_ori_img, save_img = apply_colormap_on_image(ori_img, data)
            save_img.save(save_name)

        # else:
        #     #img_list = []
        #     if not name =='after2':
        #         img_list = []
        #         for i in range(len(input)):
        #             save_dir = os.path.join(medium_results_path, base_name + '_' + name + 'input' +str(i))
        #             check_mkdir(save_dir)
        #             #save_name = os.path.join(save_dir, image_name + '.png')
        #
        #             data = input[i].clone().detach()
        #
        #             data = data.permute(1, 0, 2, 3)
        #             data = get_cam(ori_img, data.cpu())
        #             no_ori_img, save_img = apply_colormap_on_image(ori_img, data)
        #             img_list.append(cv2.cvtColor(np.array(save_img),cv2.COLOR_RGB2BGR))
        #             #save_img.save(save_name)
        #         save_dir = os.path.join(medium_results_path, base_name + '_input' + name)
        #         check_mkdir(save_dir)
        #         save_name = os.path.join(save_dir, image_name + '.png')
        #         scale = 224
        #         pre_and_gt = cv2.resize(img, (scale, scale))
        #         split_img = np.ones((scale, 16, 3)) * 255
        #         for pre_img in img_list:
        #             pre_img = cv2.resize(pre_img, (scale, scale))
        #             # pre_img = np.array(pre_img)
        #             pre_and_gt = np.concatenate([pre_and_gt, split_img, pre_img], axis=1)
        #         cv2.imwrite(save_name, pre_and_gt)
        #     img_list = []
        #     for i in range(len(output)):
        #         #print(type(output[i]))
        #         #save_dir = os.path.join(medium_results_path, base_name + '_' + name + 'output' + str(i))
        #         #check_mkdir(save_dir)
        #         #save_name = os.path.join(save_dir, image_name + '.png')
        #         if len(output) == 1:
        #             data = output.clone().detach()
        #         else:
        #             data = output[i].clone().detach()
        #         data = data.permute(1, 0, 2, 3)
        #         data = get_cam(ori_img, data.cpu())
        #         no_ori_img, save_img = apply_colormap_on_image(ori_img, data)
        #         img_list.append(cv2.cvtColor(np.array(save_img),cv2.COLOR_RGB2BGR))
        #         #save_img.save(save_name)
        #
        #     save_dir = os.path.join(medium_results_path, base_name + '_output' + name)
        #     check_mkdir(save_dir)
        #     save_name = os.path.join(save_dir, image_name + '.png')
        #     #pre_and_gt = np.array(ori_img)
        #     scale = 224
        #     #pre_and_gt = cv2.resize(img, (scale, scale))
        #     pre_and_gt = cv2.resize(img_list[0],(scale,scale)) # only output
        #     split_img = np.ones((scale, 16, 3)) * 255
        #     for pre_img in img_list: #img_list[1:]:# only output
        #         pre_img = cv2.resize(pre_img,(scale,scale))
        #         #pre_img = np.array(pre_img)
        #         pre_and_gt = np.concatenate([pre_and_gt, split_img, pre_img], axis=1)
        #     cv2.imwrite(save_name, pre_and_gt)

        else:
            global dataset_name
            print(dataset_name)
            save_dir = os.path.join(medium_results_path, base_name + '_' + name + '_' + 'output',dataset_name)
            print(save_dir)
            check_mkdir(save_dir)
            save_name_output = os.path.join(save_dir, image_name + '.png')
            data = output.clone().detach()
            data = data.permute(1, 0, 2, 3)
            data = get_cam(ori_img, data.cpu())
            no_ori_img, save_img = apply_colormap_on_image(ori_img, data)
            save_img.save(save_name_output)
            save_dir = os.path.join(medium_results_path, base_name + '_' + name + '_' + 'input', dataset_name)
            check_mkdir(save_dir)
            save_name_input = os.path.join(save_dir, image_name +'.png')
            data = input[0].clone().detach()
            data = data.permute(1, 0, 2, 3)
            data = get_cam(ori_img, data.cpu())
            no_ori_img, save_img = apply_colormap_on_image(ori_img, data)
            save_img.save(save_name_input)


    return hook_func


def main(exp_name,net,pretrained_model_path,modules_for_plot,scale = args['scale']):
    global cnt
    cnt = 0
    net.load_state_dict(torch.load(pretrained_model_path))
    net.eval()

    img_transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_resize = transforms.Compose([
        transforms.Resize((scale, scale)),
    ])

    for name, module in modules_for_plot.items():
        # global module_name
        # module_name = name
        module.register_forward_hook(get_activation(name))

    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            global dataset_name
            dataset_name = name
            time_list = []
            image_path = os.path.join(root, 'Imgs')
            if args['save_results']:
                check_mkdir(os.path.join('/results', exp_name, name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in enumerate(img_list):
                global image_name, ori_img, ori_img_path
                image_name = img_name
                ori_img_path = os.path.join(image_path, img_name  + '.jpg')
                #img = cv2.imread(os.path.join(image_path, img_name  + '.jpg'))
                img = Image.open(os.path.join(image_path, img_name  + '.jpg')).convert('RGB')
                #w, h = img.size
                ori_img = img
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

def get_features_heatmap(modelname,net,pretrained_model_path,modules_for_plot,scale):
    global medium_results_path
    medium_results_path = os.path.join(medium_results_path, modelname)
    check_mkdir(medium_results_path)
    main(modelname, net, pretrained_model_path, modules_for_plot,scale)

if __name__ == '__main__':
    from SARNet import SARNet
    net = SARNet('pvt_v2_b3').cuda()
    pth_path = os.path.join(root,'SARNet/pth/SARNet.pth')
    exp_name = 'SARNet'
    modules_for_plot = {
        'focus2': net.focus2, 'after2': net.focus2.increase_input_map
    }
    get_features_heatmap(exp_name,net,pth_path,modules_for_plot,384)