# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import yaml


def load_configs(file_path, ws_dir):
    with open(file_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    config_dict['log_dir'] = os.path.join(ws_dir, config_dict['log_dir'])
    config_dict['data']['raw_dataset_dir'] = os.path.join(ws_dir, config_dict['data']['raw_dataset_dir'])
    config_dict['data']['dataset_dir'] = os.path.join(ws_dir, config_dict['data']['dataset_dir'])
    config_dict['data']['bin_path'] = os.path.join(ws_dir, config_dict['data']['bin_path'])
    config_dict['data']['smpl_path'] = os.path.join(ws_dir, config_dict['data']['smpl_path'])
    config_dict['data']['uv_info'] = os.path.join(ws_dir, config_dict['data']['uv_info'])
    config_dict['data']['resample_idxs_path'] = os.path.join(ws_dir, config_dict['data']['resample_idxs_path'])
    config_dict['data']['train_bin_path'] = os.path.join(ws_dir, config_dict['data']['train_bin_path'])
    config_dict['data']['interp_bin_path'] = os.path.join(ws_dir, config_dict['data']['interp_bin_path'])
    config_dict['data']['extrap_bin_path'] = os.path.join(ws_dir, config_dict['data']['extrap_bin_path'])

    if 'type' not in config_dict['data']:
        config_dict['data']['type'] = 'CAPE'
    if 'separate_detail' not in config_dict['data']:
        config_dict['data']['separate_detail'] = True

    return config_dict
