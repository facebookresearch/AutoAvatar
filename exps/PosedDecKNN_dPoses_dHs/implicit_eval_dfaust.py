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
import glob
import random
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
torch.cuda.manual_seed_all(777)
random.seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


from argparse import ArgumentParser

parser = ArgumentParser(description='Test AutoAvatar.')
parser.add_argument('--ws_dir', required=True, help='path of work space directory')
parser.add_argument('--ckpt_dir', required=True, help='path of checkpoint directory')
parser.add_argument('--ckpt_itr', default=7500, type=int)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--resolution', default=256, type=int, help='marching cube resolution')
parser.add_argument('--data_mode', default='extrap', type=str, help='test which type of data. choose from ["extrap", "interp"]')
cmd_args = parser.parse_args()

gpu_id = cmd_args.gpu_id
resolution = cmd_args.resolution
data_mode = cmd_args.data_mode
eval_frames = [3] #list(range(3, 99999, 20)) #
ckpt_dir = cmd_args.ckpt_dir
#'/mnt/ImpDyn_ws/logs/Feb27_00-22-01_04s_50002_v2_PosedDecKNN_dPoses_dHs_AutoRegr_Rollout2'
#'/mnt/ImpDyn_ws/logs/May26_20-56-44_04s_50002_v2_PosedDecKNN_dPoses_dHs_HalfSub_AutoRegr_Rollout2'
#'/mnt/ImpDyn_ws/logs/Mar03_10-44-09_04s_50004_v2_PosedDecKNN_dPoses_dHs_AutoRegr_Rollout8'
#'/mnt/ImpDyn_ws/logs/Mar03_10-42-05_04s_50002_v2_PosedDecKNN_dPoses_dHs_AutoRegr_Rollout8'
#'/mnt/ImpDyn_ws/logs/Feb27_20-50-42_04s_50004_v2_PosedDecKNN_dPoses_dHs_AutoRegr_Rollout2'
#'/mnt/ImpDyn_ws/logs/Feb27_09-57-21_04s_50004_v2_PosedDecKNN_dPoses_dHs_AutoRegr'#Feb27_20-50-42_04s_50004_v2_PosedDecKNN_dPoses_dHs_AutoRegr_Rollout2'
#'/mnt/ImpDyn_ws/logs/Feb23_12-43-09_04s_50002_v2_PosedDecKNN_Dyna_dHs_AutoRegr_Rollout2'
#'/mnt/ImpDyn_ws/logs/Feb19_01-26-38_04s_50002_NA_PosedDecKNN_Dyna_Hs_AutoRegr'
ckpt_itr = cmd_args.ckpt_itr
#90000#7500
#105000
configs_path = glob.glob(os.path.join(ckpt_dir, 'net_def', '*.yaml'))[0]
args = load_configs(configs_path, cmd_args.ws_dir)
dev_tag = '72s'
subject_tag = args['data']['subject'] + '_' + args['data']['cloth_type']
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_name = current_time + '_' + dev_tag + '_' + subject_tag + '_' + args['name'] + '_' + args['version'] + '_' + data_mode
args['log_dir'] = os.path.join(cmd_args.ws_dir, 'logs_test')
args['train']['n_rollout'] = 32
with open(args['data']['interp_bin_path'], 'rb') as f:
    interp_list = pickle.load(f)
with open(args['data']['extrap_bin_path'], 'rb') as f:
    extrap_list = pickle.load(f)
if data_mode == 'extrap':
    seqs_list = extrap_list
elif data_mode == 'interp':
    seqs_list = interp_list
elif data_mode == 'train':
    seqs_list = [4]

if not os.path.exists(os.path.join(args['log_dir'], log_name)):
    os.makedirs(os.path.join(args['log_dir'], log_name))
for seq_idx in seqs_list:
    log_dir = os.path.join(args['log_dir'], log_name, 'seq_%03d' % seq_idx)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    dfaust_json = dfaust_utils.DFaustJson(args['data']['bin_path'])
    validset = DFaustDataset(args, dfaust_json, [seq_idx], eval_frames=eval_frames)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)

    logger = TensorBoardLogger(log_dir, name='')
    trainbox = Implicit_Trainbox(args, log_dir, resolution, eval_frames=eval_frames)
    if ckpt_dir is not None:
        trainbox.load_ckpt(ckpt_itr, ckpt_dir)
    shutil.copy(os.path.realpath(__file__), os.path.join(log_dir, 'net_def'))
    shutil.copy(configs_path, os.path.join(log_dir, 'net_def'))

    train_params = {
        'max_steps': 10,
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
    trainer.test(trainbox, valid_loader)
