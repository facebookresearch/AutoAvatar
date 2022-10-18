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


class DFaustDataset(Dataset):
    def __init__(self, args, dfaust_json, seq_list, skip=1, gap=1, eval_frames=None, no_mesh=False) -> None:
        super().__init__()
        self.args = copy.deepcopy(args)
        self.dfaust_json = dfaust_json
        self.seq_list = seq_list
        self.skip = skip
        self.gap = gap
        self.eval_frames = eval_frames
        self.no_mesh = no_mesh
        self.n_frames = args['model']['n_batch_frames']
        self.n_rollout = args['train']['n_rollout']
        self.raw_dataset_dir = args['data']['raw_dataset_dir']
        self.dataset_dir = args['data']['dataset_dir']
        self.smooth_tag = args['data']['smooth_tag']
        # self.faces = torch.from_numpy(np.load(os.path.join(args['data']['raw_dataset_dir'], 'misc', 'smpl_tris.npy')).astype(np.int32)).long()

        self.samples = [[], []]
        for seq_idx in seq_list:
            seq = dfaust_json.data['seqs'][seq_idx]
            seq_len = len(seq['frames'])
            if eval_frames is None:
                frame_idxs = list(range(0, seq_len - (self.n_frames - 2 + self.n_rollout) * skip, gap))
            else:
                frame_idxs = list(range(0, seq_len - self.n_frames + 1, 1))
            self.samples[0] += [seq_idx] * len(frame_idxs)
            self.samples[1] += frame_idxs

        assert len(self.samples[0]) == len(self.samples[1])

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, index):
        seq_idx, frame_idx = self.samples[0][index], self.samples[1][index]
        end_idx = frame_idx + (self.n_frames + self.n_rollout - 1) * self.skip
        if self.eval_frames is not None:
            if frame_idx + self.args['model']['n_hist_frames'] == self.eval_frames[0]:
                end_idx = len(self.dfaust_json.data['seqs'][seq_idx]['frames'])
            else:
                end_idx = min(end_idx, len(self.dfaust_json.data['seqs'][seq_idx]['frames']))

        verts_list = []
        faces_list = []
        poses_list = []
        verts_smt_list = []
        faces_smt_list = []
        z_ids_list = []
        for i in range(frame_idx, end_idx, self.skip):
            frame = self.dfaust_json.data['seqs'][seq_idx]['frames'][i]
            ply_path = os.path.join(self.raw_dataset_dir, frame['ply_path'])
            ply_path_ = os.path.join(self.dataset_dir, 'scans_simple', self.dfaust_json.data['subject'],
                                     self.dfaust_json.data['seqs'][seq_idx]['seq_name'], ply_path.split('/')[-1])
            if not self.no_mesh:
                verts, faces = load_ply(ply_path_)
            else:
                verts, faces = torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.long)
            poses = frame['poses']
            assert poses.shape == (75,)
            if not self.args['data']['separate_detail']:
                verts_smt, faces_smt = verts.clone(), faces.clone()
            else:
                ply_path_ = os.path.join(self.dataset_dir, self.smooth_tag, self.dfaust_json.data['subject'],
                                        self.dfaust_json.data['seqs'][seq_idx]['seq_name'], ply_path.split('/')[-1][:-4] + '_smt.ply')
                verts_smt, faces_smt = load_ply(ply_path_)

            z_ids_list.append(torch.tensor(frame['z_id']).long())
            verts_list.append(verts)
            faces_list.append(faces)
            poses_list.append(torch.from_numpy(poses).float())
            verts_smt_list.append(verts_smt)
            faces_smt_list.append(faces_smt)

        z_ids_list = torch.stack(z_ids_list, dim=0)
        poses_list = torch.stack(poses_list, dim=0)

        return {'verts_detail': verts_list, 'faces_detail': faces_list, 'verts_smt': verts_smt_list, 'faces_smt': faces_smt_list, 'poses': poses_list, 'z_ids': z_ids_list}
