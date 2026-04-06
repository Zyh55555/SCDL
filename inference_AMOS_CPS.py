import argparse
import glob
import logging
import os
import random
import stat
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.vnet import VNet
from utils import test_amos_vnet_AB


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='AMOS', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./data/AMOS/', help='dataset root path')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save logs and metrics')
parser.add_argument('--exp', type=str, default='CPS', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic validation')
parser.add_argument('--labelnum', type=int, default=10, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--GA', type=int, default=0, help='whether use GA loss')
parser.add_argument('--model_path_A', type=str, default='', help='checkpoint path for model A')
parser.add_argument('--model_path_B', type=str, default='', help='checkpoint path for model B')

args = parser.parse_args()


def read_list(split):
    ids_list = np.loadtxt(
        os.path.join('./data/amos_splits/', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


test_list = read_list('test')
num_classes = 16
patch_size = (96, 96, 96)
train_data_path = args.root_path

if args.GA:
    exp_name = f'{args.dataset_name}_{args.exp}_GA_{args.labelnum}labeled'
else:
    exp_name = f'{args.dataset_name}_{args.exp}_{args.labelnum}labeled'
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

    handler = logging.FileHandler(snapshot_path_tmp + '/log_{}.txt'.format(typename), mode='w')
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


def create_model(n_classes=16):
    net = VNet(n_classes=n_classes)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    return net.cuda()


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'net', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def align_state_dict_keys(state_dict, model):
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if not model_keys or not ckpt_keys:
        return state_dict

    model_has_module = model_keys[0].startswith('module.')
    ckpt_has_module = ckpt_keys[0].startswith('module.')
    if model_has_module == ckpt_has_module:
        return state_dict

    aligned_state_dict = OrderedDict()
    if ckpt_has_module and not model_has_module:
        for key, value in state_dict.items():
            aligned_state_dict[key[7:] if key.startswith('module.') else key] = value
        return aligned_state_dict

    for key, value in state_dict.items():
        aligned_state_dict[f'module.{key}'] = value
    return aligned_state_dict


def load_model_weights(model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = extract_state_dict(checkpoint)
    state_dict = align_state_dict_keys(state_dict, model)
    model.load_state_dict(state_dict)


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


def log_metrics(metric_mean, metric_std):
    class_names = [
        'spleen',
        'r.kidney',
        'l.kidney',
        'gallbladder',
        'esophagus',
        'liver',
        'stomach',
        'aorta',
        'ivc',
        'pancreas',
        'right adrenal gland',
        'left adrenal gland',
        'duodenum',
        'bladder',
        'prostate/uterus',
    ]

    logging.info(
        'Final Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}'.format(
            metric_mean[0].mean(),
            metric_mean[1].mean(),
            metric_mean[2].mean(),
            metric_mean[3].mean(),
        )
    )

    for class_idx, class_name in enumerate(class_names):
        logging.info(
            '{}: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}'.format(
                class_name,
                metric_mean[0][class_idx], metric_std[0][class_idx],
                metric_mean[1][class_idx], metric_std[1][class_idx],
                metric_mean[2][class_idx], metric_std[2][class_idx],
                metric_mean[3][class_idx], metric_std[3][class_idx],
            )
        )


def run_validation(model_path_A, model_path_B):
    model_A = create_model(n_classes=num_classes)
    load_model_weights(model_A, model_path_A)
    model_A.eval()

    model_B = create_model(n_classes=num_classes)
    load_model_weights(model_B, model_path_B)
    model_B.eval()

    _, _, metric_final = test_amos_vnet_AB.validation_all_case(
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
    log_metrics(metric_mean, metric_std)
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPUs are available')

    prepare_output_dir()
    best_model_path_A, best_model_path_B = resolve_validation_checkpoints()
    print('snapshot_path:', snapshot_path)
    print('model_path_A:', best_model_path_A)
    print('model_path_B:', best_model_path_B)
    run_validation(best_model_path_A, best_model_path_B)
