import argparse
import glob
import logging
import os
import random
import stat
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.vnet import VNet
from utils import test_util_vnet_AB


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Synapse', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./data/Synapse/', help='dataset root path')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save logs and metrics')
parser.add_argument('--exp', type=str, default='CPS_SCDL', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic validation')
parser.add_argument('--labelnum', type=int, default=4, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--model_path_A', type=str, default='', help='checkpoint path for model A')
parser.add_argument('--model_path_B', type=str, default='', help='checkpoint path for model B')

args = parser.parse_args()


test_list = ['0004', '0007', '0010', '0033', '0035', '0036']
num_classes = 14
patch_size = (96, 96, 96)
train_data_path = args.root_path
exp_name = f"{args.dataset_name}_{args.exp}_GA_{args.labelnum}labeled_seed_{args.seed}"
snapshot_path = os.path.abspath(os.path.join(args.save_path, exp_name))


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


def config_log(snapshot_path_tmp, typename):
    formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(snapshot_path_tmp + "/log_{}.txt".format(typename), mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def prepare_output_dir():
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.chmod(snapshot_path, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)


def create_model(n_classes=14):
    net = VNet(n_channels=1, n_classes=n_classes)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    return net.cuda()


def resolve_validation_checkpoints():
    model_path_A = args.model_path_A.strip()
    model_path_B = args.model_path_B.strip()

    if bool(model_path_A) != bool(model_path_B):
        raise ValueError('Please provide both --model_path_A and --model_path_B, or leave both empty.')

    if model_path_A and model_path_B:
        model_path_A = os.path.abspath(model_path_A)
        model_path_B = os.path.abspath(model_path_B)
        if not os.path.exists(model_path_A):
            raise FileNotFoundError('Model A checkpoint not found: {}'.format(model_path_A))
        if not os.path.exists(model_path_B):
            raise FileNotFoundError('Model B checkpoint not found: {}'.format(model_path_B))
        return model_path_A, model_path_B

    best_model_candidates = sorted(
        glob.glob(os.path.join(snapshot_path, '*_best_A.pth')),
        key=os.path.getmtime,
        reverse=True
    )
    for candidate_A in best_model_candidates:
        candidate_B = candidate_A.replace('_best_A.pth', '_best_B.pth')
        if os.path.exists(candidate_B):
            return candidate_A, candidate_B

    raise FileNotFoundError(
        'No paired best checkpoints found under {}. Please pass --model_path_A and --model_path_B.'.format(snapshot_path)
    )


def run_validation(model_path_A, model_path_B):
    model_A = create_model(n_classes=num_classes)
    model_A.load_state_dict(torch.load(model_path_A))
    model_A.eval()

    model_B = create_model(n_classes=num_classes)
    model_B.load_state_dict(torch.load(model_path_B))
    model_B.eval()

    _, _, metric_final = test_util_vnet_AB.validation_all_case(
        model_A,
        model_B,
        num_classes=num_classes,
        base_dir=train_data_path,
        image_list=test_list,
        patch_size=patch_size,
        stride_xy=32,
        stride_z=16
    )

    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)
    metric_save_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_save_path, metric_final)

    handler, sh = config_log(snapshot_path, 'total_metric')
    logging.info('Validation checkpoint A: {}'.format(model_path_A))
    logging.info('Validation checkpoint B: {}'.format(model_path_B))
    logging.info('Final Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}, \n'
                 'spleen: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'r.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'l.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'gallbladder: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'esophagus: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'liver: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'stomach: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'aorta: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'ivc: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'portal and splenic vein: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'pancreas: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'right adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'Left adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}'
                 .format(metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(), metric_mean[3].mean(),
                         metric_mean[0][0], metric_std[0][0], metric_mean[1][0], metric_std[1][0], metric_mean[2][0], metric_std[2][0], metric_mean[3][0], metric_std[3][0],
                         metric_mean[0][1], metric_std[0][1], metric_mean[1][1], metric_std[1][1], metric_mean[2][1], metric_std[2][1], metric_mean[3][1], metric_std[3][1],
                         metric_mean[0][2], metric_std[0][2], metric_mean[1][2], metric_std[1][2], metric_mean[2][2], metric_std[2][2], metric_mean[3][2], metric_std[3][2],
                         metric_mean[0][3], metric_std[0][3], metric_mean[1][3], metric_std[1][3], metric_mean[2][3], metric_std[2][3], metric_mean[3][3], metric_std[3][3],
                         metric_mean[0][4], metric_std[0][4], metric_mean[1][4], metric_std[1][4], metric_mean[2][4], metric_std[2][4], metric_mean[3][4], metric_std[3][4],
                         metric_mean[0][5], metric_std[0][5], metric_mean[1][5], metric_std[1][5], metric_mean[2][5], metric_std[2][5], metric_mean[3][5], metric_std[3][5],
                         metric_mean[0][6], metric_std[0][6], metric_mean[1][6], metric_std[1][6], metric_mean[2][6], metric_std[2][6], metric_mean[3][6], metric_std[3][6],
                         metric_mean[0][7], metric_std[0][7], metric_mean[1][7], metric_std[1][7], metric_mean[2][7], metric_std[2][7], metric_mean[3][7], metric_std[3][7],
                         metric_mean[0][8], metric_std[0][8], metric_mean[1][8], metric_std[1][8], metric_mean[2][8], metric_std[2][8], metric_mean[3][8], metric_std[3][8],
                         metric_mean[0][9], metric_std[0][9], metric_mean[1][9], metric_std[1][9], metric_mean[2][9], metric_std[2][9], metric_mean[3][9], metric_std[3][9],
                         metric_mean[0][10], metric_std[0][10], metric_mean[1][10], metric_std[1][10], metric_mean[2][10], metric_std[2][10], metric_mean[3][10], metric_std[3][10],
                         metric_mean[0][11], metric_std[0][11], metric_mean[1][11], metric_std[1][11], metric_mean[2][11], metric_std[2][11], metric_mean[3][11], metric_std[3][11],
                         metric_mean[0][12], metric_std[0][12], metric_mean[1][12], metric_std[1][12], metric_mean[2][12], metric_std[2][12], metric_mean[3][12], metric_std[3][12]))
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPUs are available')

    prepare_output_dir()
    best_model_path_A, best_model_path_B = resolve_validation_checkpoints()
    print('snapshot_path:', snapshot_path)
    print('model_path_A:', best_model_path_A)
    print('model_path_B:', best_model_path_B)
    run_validation(best_model_path_A, best_model_path_B)
