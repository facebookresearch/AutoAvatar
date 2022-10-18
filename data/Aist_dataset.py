# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import copy
import pickle
import torch
from torch.utils.data import Dataset
from pytorch3d.io import load_ply

import utils.CAPE as cape_utils


class AistDataset(Dataset):
    def __init__(self, args, dfaust_json, seq_dir, skip=1, gap=1, eval_frames=None) -> None:
        super().__init__()
        self.args = copy.deepcopy(args)
        self.dfaust_json = dfaust_json
        self.seq_dir = seq_dir
        self.skip = skip
        self.gap = gap
        self.eval_frames = eval_frames
        self.n_frames = args['model']['n_batch_frames']
        self.n_rollout = args['train']['n_rollout']
        self.raw_dataset_dir = args['data']['raw_dataset_dir']
        self.dataset_dir = args['data']['dataset_dir']
        self.smooth_tag = args['data']['smooth_tag']
        # self.faces = torch.from_numpy(np.load(os.path.join(args['data']['raw_dataset_dir'], 'misc', 'smpl_tris.npy')).astype(np.int32)).long()

        self.frame_list = sorted(os.listdir(seq_dir))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        verts_list = []
        faces_list = []
        poses_list = []
        verts_smt_list = []
        faces_smt_list = []
        for i in range(index, len(self.frame_list), self.skip):
            data = np.load(os.path.join(self.seq_dir, self.frame_list[i]))
            rot, transl = data['pose'], data['transl']
            poses = np.concatenate([transl, rot], axis=0)
            assert poses.shape == (75,)
            poses_list.append(torch.from_numpy(poses).float())

        poses_list = torch.stack(poses_list, dim=0)

        frame = self.dfaust_json.data['seqs'][0]['frames'][0]
        ply_path = os.path.join(self.raw_dataset_dir, frame['ply_path'])
        ply_path_ = os.path.join(self.dataset_dir, 'scans_simple_2nd', self.dfaust_json.data['subject'],
                                 self.dfaust_json.data['seqs'][0]['seq_name'], ply_path.split('/')[-1])
        verts_init, faces_init = load_ply(ply_path_)
        poses_init = torch.from_numpy(frame['poses']).float()
        assert poses_init.shape == (75,)

        z_ids_list = torch.zeros(poses_list.shape[0]).long()

        return {'verts_detail': verts_list, 'faces_detail': faces_list, 'verts_smt': verts_smt_list, 'faces_smt': faces_smt_list, 'poses': poses_list,
                'verts_init': verts_init, 'faces_init': faces_init, 'poses_init': poses_init, 'z_ids': z_ids_list}
