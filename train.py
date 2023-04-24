import shutil
import sys
import numpy as np
import datetime
import time
import os
from tqdm import tqdm
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import joint_transforms
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
import loss
import cv2

cudnn.benchmark = True

def norm_img(im):
    return cv2.normalize(im.astype('float'),
                         None,
                         0.0, 1.0,
                         cv2.NORM_MINMAX)

def read_and_normalize(gt_img, sm_img, gt_threshold=0.5):
    """
    function that reads, normalizes and crops a ground truth and a saliency map

    parameters
    ----------
    gt_threshold : float
        The threshold that is used to binrize ground truth maps.

    Returns
    -------
    gt_img, sm_img : numpy.ndarray
        The prepared arrays
    """
    gt_img = norm_img(gt_img)
    gt_img = (gt_img >= gt_threshold).astype(np.float32)
    sm_img = norm_img(sm_img)
    if sm_img.shape[0] != gt_img.shape[0] or sm_img.shape[1] != gt_img.shape[1]:
        sm_img = cv2.resize(sm_img, (gt_img.shape[1], gt_img.shape[0]))
    return gt_img, sm_img

def move_all_files(src,dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    for file in os.listdir(src):
        full_file_name = os.path.join(src, file)  # 把文件的完整路径得到
        if os.path.isfile(full_file_name):  # 用于判断某一对象(需提供绝对路径)是否为文件
            shutil.copy(full_file_name, dst)  # shutil.copy函数放入原文件的路径文件全名  然后放入目标文件夹
    return

def size_format(b):
    size = 1024
    if b < size:
        return '%i' % b + 'B'
    elif size <= b < np.power(size,2):
        return '%.2f' % float(b/size) + 'KB'
    elif np.power(size,2) <= b < np.power(size,3):
        return '%.2f' % float(b/np.power(size,2)) + 'MB'
    elif np.power(size,3) <= b < np.power(size,4):
        return '%.2f' % float(b/np.power(size,3)) + 'GB'
    elif np.power(size,4) <= b:
        return '%.2f' % float(b/np.power(size,4)) + 'TB'

def print_network(model, name):  # 1M = 10^6
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    #print(model)
    open(log_path, 'w').write("The number of {} parameters: {}".format(name,size_format(num_params)) + '\n\n')
    print("The number of {} parameters: {}".format(name,size_format(num_params)))
from SARNet import SARNet
import config
ckpt_path = './ckpt'
exp_name = 'SARNet_v1'
pvt_name = 'pvt_v2_b3'
save_epoch_num = 100
args = {
    'epoch_num': save_epoch_num,
    'train_batch_size': 2,
    'last_epoch': 0,
    'lr': 1e-3, #1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 384, #224,
    'save_point': [50, 60, 90, 'best'],
    'poly_train': True,
    'optimizer': 'SGD',
    'save_predict_images': True
}

to_pil = transforms.ToPILImage()

print("pytroch的版本", torch.__version__)
# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), #亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
predict_img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_transform = transforms.ToTensor()
cod_training_root = config.cod_training_root
train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)  # 计算一共需要的次数

# loss function
structure_loss = loss.structure_loss().cuda()
bce_loss = nn.BCEWithLogitsLoss().cuda( )
iou_loss = loss.IOU().cuda()

open(log_path, 'w').write('IOU' + '\n\n')
#net
open(log_path, 'w').write(str(exp_name) + '\n\n')
open(log_path, 'w').write(str(args) + '\n\n')
print(args)
print(exp_name)


def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def main(net):

    print_network(net,exp_name)
    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])


    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        print('load pretrain model')
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    net = nn.DataParallel(net,)
    print("Using {} GPU(s) to Train.".format(os.environ['CUDA_VISIBLE_DEVICES']))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

def train(net, optimizer):
    min_mae = 1
    curr_iter = 1
    start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        net.train()
        loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record ,loss_5_record= AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr
            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda( )
            labels = Variable(labels).cuda( )
            optimizer.zero_grad()
            predict_1, predict_2, predict_3, predict_4, predict_5 = net(inputs)


            loss_1 = structure_loss(predict_1, labels)
            loss_2 = structure_loss(predict_2, labels)
            loss_3 = structure_loss(predict_3, labels)
            loss_4 = structure_loss(predict_4, labels)
            loss_5 = structure_loss(predict_5, labels)

            loss = 1 * loss_1 + 2 * loss_2 + 2 * loss_3 + 3 * loss_4 + 6 * loss_5

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_5_record.update(loss_5.data, batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('base_lr', base_lr, curr_iter)
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)
                writer.add_scalar('loss_5', loss_5, curr_iter)

            log = '[%d], [%d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg,
                   loss_3_record.avg, loss_4_record.avg,loss_5_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

            tmp_path = './tem_see'
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)

        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda()

        if epoch > args['epoch_num']:
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            return


if __name__ == '__main__':
    net = SARNet(pvt_name)
    net = net.cuda().train()
    net = torch.nn.DataParallel(net)
    main(net)
    results_path = './results'
    from new_infer import evaluation_COD
    for i in args['save_point']:
      pth_path = os.path.join(ckpt_path, exp_name, '%d.pth' % i)
      evaluation_COD(exp_name,net,args['scale'],results_path,pth_path)
