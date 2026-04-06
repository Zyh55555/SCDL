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
import torch.nn as nn
import torch.nn.functional as F

from networks.SCDL import SCDL
from networks.magicnet import VNet_Magic
from utils import test_util


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Synapse', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./data/Synapse', help='dataset root path')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save logs and metrics')
parser.add_argument('--exp', type=str, default='MagicNet', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic validation')
parser.add_argument('--labelnum', type=int, default=4, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')
parser.add_argument('--batch_size', type=int, default=4, help='batch size used to initialize SCDL')
parser.add_argument('--model_path', type=str, default='', help='combined checkpoint path or model checkpoint path')
parser.add_argument('--eprl_path', type=str, default='', help='optional separate checkpoint path for SCDL')

args = parser.parse_args()

tok_size = 12
eprl_zdim = 256
num_classes = 14
patch_size = (96, 96, 96)
train_data_path = args.root_path
test_list = ['0004', '0007', '0010', '0033', '0035', '0036']
snapshot_path = os.path.abspath(
    os.path.join(args.save_path, '{}_{}_GA_{}labeled_seed_{}'.format(
        args.dataset_name, args.exp, args.labelnum, args.seed
    ))
)


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


def create_model(n_classes=14, cube_size=32, patchsize=96):
    net = VNet_Magic(n_channels=1, n_classes=n_classes, cube_size=cube_size, patch_size=patchsize)
    return net.cuda()


def create_eprl(num_classes=14):
    return SCDL(
        x_dim=16,
        z_dim=eprl_zdim,
        num_classes=num_classes,
        seed=args.seed,
        batch_size=args.batch_size
    ).cuda()


def extract_state_dict(checkpoint, key_candidates=None):
    if key_candidates is None:
        key_candidates = ('state_dict', 'model_state_dict', 'net', 'model')
    if isinstance(checkpoint, dict):
        for key in key_candidates:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def align_state_dict_keys(state_dict, module):
    module_keys = list(module.state_dict().keys())
    state_keys = list(state_dict.keys())
    if not module_keys or not state_keys:
        return state_dict

    module_has_prefix = module_keys[0].startswith('module.')
    state_has_prefix = state_keys[0].startswith('module.')
    if module_has_prefix == state_has_prefix:
        return state_dict

    aligned_state_dict = OrderedDict()
    if state_has_prefix and not module_has_prefix:
        for key, value in state_dict.items():
            aligned_state_dict[key[7:] if key.startswith('module.') else key] = value
        return aligned_state_dict

    for key, value in state_dict.items():
        aligned_state_dict['module.{}'.format(key)] = value
    return aligned_state_dict


def load_module_weights(module, state_dict):
    aligned_state_dict = align_state_dict_keys(state_dict, module)
    module.load_state_dict(aligned_state_dict, strict=False)


def resolve_validation_checkpoint():
    model_path = args.model_path.strip()
    if model_path:
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError('Checkpoint not found: {}'.format(model_path))
        return model_path

    best_model_candidates = sorted(
        glob.glob(os.path.join(snapshot_path, '*_best.pth')),
        key=os.path.getmtime,
        reverse=True
    )
    if best_model_candidates:
        return best_model_candidates[0]

    raise FileNotFoundError(
        'No best checkpoint found under {}. Please pass --model_path.'.format(snapshot_path)
    )


def compute_prior_map_from_embedding(embedding, eprl, tok_size=12, z_dim=256):
    emb_ds = F.adaptive_avg_pool3d(embedding, output_size=(tok_size, tok_size, tok_size))
    x_tok = emb_ds.flatten(2).permute(0, 2, 1).contiguous()
    _, prior_tok, *_ = eprl(x_tok, y_dist_flat=None)
    prior_map = prior_tok.permute(0, 2, 1).contiguous().view(
        embedding.size(0), z_dim, tok_size, tok_size, tok_size
    )
    return prior_map


class VNetWithPrior(nn.Module):
    def __init__(self, base_model, eprl, num_classes, tok_size=12, z_dim=256):
        super().__init__()
        self.base = base_model
        self.eprl = eprl
        self.num_classes = num_classes
        self.tok_size = tok_size
        self.z_dim = z_dim

    def forward(self, x):
        _, emb = self.base(x)
        prior_map = compute_prior_map_from_embedding(
            emb,
            self.eprl,
            tok_size=self.tok_size,
            z_dim=self.z_dim
        )
        out = self.base.forward_prediction_head(emb, prior_map=prior_map)
        return out, emb


def load_checkpoint_weights(model, eprl, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state_dict = extract_state_dict(checkpoint, key_candidates=('model', 'state_dict', 'model_state_dict', 'net'))
        load_module_weights(model, model_state_dict)

        if 'eprl' in checkpoint and isinstance(checkpoint['eprl'], dict):
            load_module_weights(eprl, checkpoint['eprl'])
            return

        if args.eprl_path:
            eprl_checkpoint = torch.load(os.path.abspath(args.eprl_path), map_location='cpu')
            eprl_state_dict = extract_state_dict(eprl_checkpoint, key_candidates=('eprl', 'state_dict', 'model_state_dict', 'net', 'model'))
            load_module_weights(eprl, eprl_state_dict)
            return

        raise KeyError('Checkpoint {} does not contain `eprl` weights. Please pass --eprl_path.'.format(checkpoint_path))

    load_module_weights(model, extract_state_dict(checkpoint))
    if not args.eprl_path:
        raise KeyError('Checkpoint {} only contains model weights. Please also pass --eprl_path.'.format(checkpoint_path))

    eprl_checkpoint = torch.load(os.path.abspath(args.eprl_path), map_location='cpu')
    eprl_state_dict = extract_state_dict(eprl_checkpoint, key_candidates=('eprl', 'state_dict', 'model_state_dict', 'net', 'model'))
    load_module_weights(eprl, eprl_state_dict)


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
        'portal and splenic vein',
        'pancreas',
        'right adrenal gland',
        'left adrenal gland',
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


def run_validation(checkpoint_path):
    model = create_model(n_classes=num_classes, cube_size=args.cube_size, patchsize=patch_size[0])
    eprl = create_eprl(num_classes=num_classes)
    load_checkpoint_weights(model, eprl, checkpoint_path)

    model.eval()
    eprl.eval()
    model_prior = VNetWithPrior(model, eprl, num_classes, tok_size=tok_size, z_dim=eprl_zdim)

    _, _, metric_final = test_util.validation_all_case(
        model_prior,
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
    logging.info('Validation checkpoint: {}'.format(checkpoint_path))
    if args.eprl_path:
        logging.info('SCDL checkpoint override: {}'.format(os.path.abspath(args.eprl_path)))
    log_metrics(metric_mean, metric_std)
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPUs are available')

    prepare_output_dir()
    best_model_path = resolve_validation_checkpoint()
    print('snapshot_path:', snapshot_path)
    print('model_path:', best_model_path)
    run_validation(best_model_path)
