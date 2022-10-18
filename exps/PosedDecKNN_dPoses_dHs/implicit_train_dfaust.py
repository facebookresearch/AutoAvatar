# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import pickle
import datetime
import shutil
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from utils.configs import *
from utils.io import *
import utils.CAPE as cape_utils
import utils.DFaust as dfaust_utils
from data.DFaust_dataset import DFaustDataset
from models.PosedDecKNN_dPoses_dHs.trainbox import Implicit_Trainbox

np.random.rand(777)
torch.random.manual_seed(777)


def train(configs_path, args, log_name, gpu_id, resolution, max_steps, ckpt_dir, ckpt_itr, ShapeEncDec_ckpt_dir, ShapeEncDec_ckpt_itr, coarse_ckpt_dir=None, coarse_ckpt_itr=None):
    dfaust_json = dfaust_utils.DFaustJson(args['data']['bin_path'])
    with open(args['data']['train_bin_path'], 'rb') as f:
        train_list = pickle.load(f)
    with open(args['data']['interp_bin_path'], 'rb') as f:
        interp_list = pickle.load(f)
    with open(args['data']['extrap_bin_path'], 'rb') as f:
        extrap_list = pickle.load(f)
    trainset = DFaustDataset(args, dfaust_json, train_list)
    validset = DFaustDataset(args, dfaust_json, extrap_list + interp_list, gap=10)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)

    logger = TensorBoardLogger(args['log_dir'], name=log_name)

    log_dir = os.path.join(args['log_dir'], log_name)
    trainbox = Implicit_Trainbox(args, log_dir, resolution)
    if ckpt_dir is not None:
        trainbox.load_ckpt(ckpt_itr, ckpt_dir)
    if ShapeEncDec_ckpt_dir is not None:
        load_components(trainbox.dyn_net, ShapeEncDec_ckpt_dir, ShapeEncDec_ckpt_itr, 'shape_enc_dec')
    if coarse_ckpt_dir is not None:
        load_components(trainbox.dyn_net, coarse_ckpt_dir, coarse_ckpt_itr, 'shape_enc_dec')
        load_components(trainbox.dyn_net, coarse_ckpt_dir, coarse_ckpt_itr, 'dynamics_net')

    shutil.copy(os.path.realpath(__file__), os.path.join(log_dir, 'net_def'))
    shutil.copy(configs_path, os.path.join(log_dir, 'net_def'))

    train_params = {
        'max_steps': max_steps,
        'gpus': [gpu_id],
        'logger': logger,
        'max_epochs': 200000,
        'log_every_n_steps': 50,
    }
    if 'check_val_every_n_epoch' in args['train']:
        train_params['check_val_every_n_epoch'] = args['train']['check_val_every_n_epoch']
    else:
        train_params['val_check_interval'] = args['train']['ckpt_step']
    trainer = Trainer(**train_params)
    trainer.fit(trainbox, train_loader, valid_loader)


from argparse import ArgumentParser

parser = ArgumentParser(description='Train AutoAvatar.')
parser.add_argument('--ws_dir', required=True, help='path of work space directory')
parser.add_argument('--configs_path', required=True, help='path of configs file')
parser.add_argument('--configs_path_rollout', required=True, help='path of configs file')
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--resolution', default=128, type=int, help='marching cube resolution')
parser.add_argument('--max_steps', default=90000, type=int, help='max training steps')
parser.add_argument('--max_steps_rollout', default=7500, type=int, help='max training steps')
cmd_args = parser.parse_args()

ShapeEncDec_ckpt_dir = None
ShapeEncDec_ckpt_itr = None
gpu_id = cmd_args.gpu_id
resolution = cmd_args.resolution
max_steps = cmd_args.max_steps
configs_path = cmd_args.configs_path
args = load_configs(configs_path, cmd_args.ws_dir)
dev_tag = '04s'
subject_tag = args['data']['subject'] + '_' + args['data']['cloth_type']
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_name = current_time + '_' + dev_tag + '_' + subject_tag + '_' + args['name'] + '_' + args['version']
log_dir = os.path.join(args['log_dir'], log_name)

train(configs_path, args, log_name, gpu_id, resolution, max_steps + 5, None, None, ShapeEncDec_ckpt_dir, ShapeEncDec_ckpt_itr)


ckpt_dir = log_dir
ckpt_itr = max_steps
gpu_id = cmd_args.gpu_id
resolution = cmd_args.resolution
max_steps = cmd_args.max_steps_rollout
configs_path = cmd_args.configs_path_rollout
args = load_configs(configs_path, cmd_args.ws_dir)
dev_tag = '04s'
subject_tag = args['data']['subject'] + '_' + args['data']['cloth_type']
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_name = current_time + '_' + dev_tag + '_' + subject_tag + '_' + args['name'] + '_' + args['version']
log_dir = os.path.join(args['log_dir'], log_name)

train(configs_path, args, log_name, gpu_id, resolution, max_steps + 5, ckpt_dir, ckpt_itr, None, None)
