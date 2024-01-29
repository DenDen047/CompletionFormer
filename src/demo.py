"""
    CompletionFormer
    ======================================================================

    main script for training and testing.
"""


from config import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
torch.autograd.set_detect_anomaly(True)

import utility
from model.completionformer import CompletionFormer
from summary.cfsummary import CompletionFormerSummary
from metric.cfmetric import CompletionFormerMetric
from data import get as get_data
from loss.l1l2loss import L1L2Loss

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import sys
from pprint import pprint
from PIL import Image
import h5py
import torchvision.transforms as T
import matplotlib.pyplot as plt
from copy import deepcopy
from summary import cfsummary


# Minimize randomness
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume

    return new_args


# https://github.com/pytorch/vision/issues/2194
class ToNumpy:
    def __call__(self, sample):
        return np.array(sample)



def get_sparse_depth(dep, num_sample):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    idx_sample = torch.randperm(num_idx)[:num_sample]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel*height*width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    return dep_sp


def visualize_result(sample, output):
    # parameters
    cmap = 'jet'
    cm = plt.get_cmap(cmap)

    # ImageNet normalization
    img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    with torch.no_grad():
        # Parse data
        feat_init = output['pred_init']
        list_feat = deepcopy(output['pred_inter'])

        rgb = sample['rgb'].detach()
        dep = sample['dep'].detach()
        pred = output['pred'].detach()
        gt = sample['gt'].detach()

        pred = torch.clamp(pred, min=0)

        # Un-normalization
        rgb.mul_(img_std.type_as(rgb)).add_(img_mean.type_as(rgb))

        rgb = rgb[0, :, :, :].data.cpu().numpy()
        dep = dep[0, 0, :, :].data.cpu().numpy()
        pred = pred[0, 0, :, :].data.cpu().numpy()
        pred_gray = pred
        gt = gt[0, 0, :, :].data.cpu().numpy()
        feat_init = feat_init[0, 0, :, :].data.cpu().numpy()
        max_depth = max(gt.max(), pred.max())
        norm = plt.Normalize(vmin=gt.min(), vmax=gt.max())

        rgb = np.transpose(rgb, (1, 2, 0))
        for k in range(0, len(list_feat)):
            feat_inter = deepcopy(list_feat[k])
            feat_inter = feat_inter[0, 0, :, :].data.cpu().numpy()
            feat_inter = np.concatenate((
                rgb,
                cm(norm(pred))[...,:3],
                cm(norm(gt))[...,:3],
                cfsummary.depth_err_to_colorbar(feat_inter, gt)
            ), axis=0)

            list_feat[k] = feat_inter

        path_output = '.'
        os.makedirs(path_output, exist_ok=True)

        path_save_rgb = '{}/01_rgb.png'.format(path_output)
        path_save_dep = '{}/02_dep.png'.format(path_output)
        path_save_init = '{}/03_pred_init.png'.format(path_output)
        path_save_pred = '{}/05_pred_final.png'.format(path_output)
        path_save_pred_gray = '{}/05_pred_final_gray.png'.format(path_output)
        path_save_gt = '{}/06_gt.png'.format(path_output)
        path_save_error = '{}/07_error.png'.format(path_output)

        plt.imsave(path_save_rgb, rgb, cmap=cmap)
        plt.imsave(path_save_gt, cm(norm(gt)))
        plt.imsave(path_save_pred, cm(norm(pred)))
        plt.imsave(path_save_pred_gray, pred_gray, cmap='gray')
        plt.imsave(path_save_dep, cm(norm(dep)))
        plt.imsave(path_save_init, cm(norm(feat_init)))
        plt.imsave(path_save_error, cfsummary.depth_err_to_colorbar(pred, gt, with_bar=True))

        for k in range(0, len(list_feat)):
            path_save_inter = '{}/04_pred_prop_{:02d}.png'.format(
                path_output, k)
            plt.imsave(path_save_inter, list_feat[k])


def test(args):
    # parameters for NYU Depth V2
    height, width = (240, 320)
    crop_size = (228, 304)
    K = torch.Tensor([
        5.1885790117450188e+02 / 2.0,
        5.1946961112127485e+02 / 2.0,
        3.2558244941119034e+02 / 2.0 - 8.0,
        2.5373616633400465e+02 / 2.0 - 6.0
    ])

    # Prepare data
    data_fpath = os.path.join(args.dir_data, 'val/official/00001.h5')
    f = h5py.File(data_fpath, 'r')
    rgb_h5 = f['rgb'][:].transpose(1, 2, 0) # [H, W, C], 0--255
    dep_h5 = f['depth'][:]  # [H, W], 1.7985953--3.615639

    rgb_img = Image.fromarray(rgb_h5, mode='RGB')
    dep_img = Image.fromarray(dep_h5.astype('float32'), mode='F')

    t_rgb = T.Compose([
        T.Resize(height),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    t_dep = T.Compose([
        T.Resize(height),
        T.CenterCrop(crop_size),
        ToNumpy(),
        T.ToTensor()
    ])

    rgb = t_rgb(rgb_img)
    dep = t_dep(dep_img)

    # make the sparse depth map
    num_sample = args.num_sample
    if num_sample < 1:
        dep_sp = torch.zeros_like(dep)
    else:
        dep_sp = get_sparse_depth(dep, num_sample)

    rgb = rgb.unsqueeze(0)
    dep = dep.unsqueeze(0)
    dep_sp = dep_sp.unsqueeze(0)

    sample = {
        'rgb': rgb,
        'dep': dep_sp,
        'gt': dep,
        'K': K
    }

    # Network
    if args.model == 'CompletionFormer':
        net = CompletionFormer(args)
    else:
        raise TypeError(args.model, ['CompletionFormer',])
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        # load the pretrained model, which might take a long time.
        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError
        print('Checkpoint loaded from {}!'.format(args.pretrain))

    net = nn.DataParallel(net)
    net.eval()
    metric = CompletionFormerMetric(args)

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/test', exist_ok=True)
    except OSError:
        pass

    writer_test = CompletionFormerSummary(args.save_dir, 'test', args, None, metric.metric_name)

    init_seed()

    # prediction
    sample = {
        key: val.cuda()
        for key, val in sample.items() if val is not None
    }
    t0 = time.time()
    with torch.no_grad():
        output = net(sample)
    t1 = time.time()
    t_total = (t1 - t0)
    print(f'processing time: {t_total} sec')

    # measure the performance
    metric_val = metric.evaluate(sample, output, 'test')
    print(f'loss: {metric_val}')

    # visualize the result
    visualize_result(sample, output)


def main(args):
    init_seed()
    test(args)


if __name__ == '__main__':
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)