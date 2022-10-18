# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import torch
import pickle
from pytorch3d.io import load_ply


# Classes -------------------------------------------------------------------------------------------------------
class DFaustJson():
    """
    DFaust .bin Structure:
        'subject'
        'seqs'
        for seq in seqs:
            'id'
            'seq_name'
            'frames'
            for frame in frames:
                'ply_path'
                'poses'
    """

    def __init__(self, bin_path=None):
        self.data = None
        if bin_path is not None:
            self.load_bin_file(bin_path)

    def load_bin_file(self, bin_path):
        with open(bin_path, 'rb') as f:
            self.data = pickle.load(f)

    def dump_bin_file(self, bin_path):
        with open(bin_path, 'wb') as f:
            pickle.dump(self.data, f)

    def append_frames(self, frames, ply_path, poses):
        frames.append({
            'ply_path': ply_path,
            'poses': poses
        })
        return frames

    def append_seqs(self, seqs, seq_name, frames):
        seqs.append({
            'id': len(seqs),
            'seq_name': seq_name,
            'frames': frames
        })
        return seqs

    def set_data(self, subject, seqs):
        self.data = {
            'subject': subject,
            'seqs': seqs
        }

    def num_of_seqs(self):
        return len(self.data['seqs'])
    
    def num_of_frames(self):
        count = 0
        for seq in self.data['seqs']:
            count += len(seq['frames'])
        return count
