# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch


def load_components(model, ckpt_dir, ckpt_itr, name):
    state_dict = model.state_dict()
    ckpt_state_dict = torch.load(os.path.join(ckpt_dir, 'ckpt', 'dyn_net_%06d.pth' % ckpt_itr), map_location='cpu')
    ckpt_state_dict = {key: value for key, value in ckpt_state_dict.items() if name in key}
    state_dict.update(ckpt_state_dict)
    model.load_state_dict(state_dict)
