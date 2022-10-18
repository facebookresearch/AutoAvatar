# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_ply
from pytorch3d.renderer import look_at_view_transform

from utils.render import render_mesh


def render_meshes(data_dir, start_i=3, gpu_id=0, simplify_mesh=True):
    if not os.path.exists(os.path.join(data_dir, 'gt_imgs')):
        os.mkdir(os.path.join(data_dir, 'gt_imgs'))
    if not os.path.exists(os.path.join(data_dir, 'pred_imgs')):
        os.mkdir(os.path.join(data_dir, 'pred_imgs'))
    # if not os.path.exists(os.path.join(data_dir, 'pred_cano_imgs')):
    #     os.mkdir(os.path.join(data_dir, 'pred_cano_imgs'))
    if not os.path.exists(os.path.join(data_dir, 'errs_imgs')):
        os.mkdir(os.path.join(data_dir, 'errs_imgs'))

    # pred_names = sorted(os.listdir(os.path.join(data_dir, 'pred_cano')))
    # for i, pred_name in enumerate(pred_names):
    #     verts, faces = load_ply(os.path.join(data_dir, 'pred_cano', pred_name))
    #     if i == 0:
    #         center = verts.median(dim=0)[0]
    #         t = center.clone()
    #         t[2] += 9
    #         R, t = look_at_view_transform(eye=t[None], at=center[None])
    #     image = render_mesh(verts.cuda(gpu_id), faces.cuda(gpu_id), R[0].cuda(gpu_id), t[0].cuda(gpu_id), 9, simplify_mesh=simplify_mesh)
    #     plt.imsave(os.path.join(data_dir, 'pred_cano_imgs', '%06d.jpg' % (i + start_i)), image.cpu().numpy())

    pred_names = sorted(os.listdir(os.path.join(data_dir, 'pred')))
    for i, pred_name in enumerate(pred_names):
        verts, faces = load_ply(os.path.join(data_dir, 'pred', pred_name))
        if i == 0:
            center = verts.median(dim=0)[0]
            t = center.clone()
            t[2] += 9
            R, t = look_at_view_transform(eye=t[None], at=center[None])
        image = render_mesh(verts.cuda(gpu_id), faces.cuda(gpu_id), R[0].cuda(gpu_id), t[0].cuda(gpu_id), 9, simplify_mesh=simplify_mesh)
        plt.imsave(os.path.join(data_dir, 'pred_imgs', '%06d.jpg' % (i + start_i)), image.cpu().numpy())

    gt_names = sorted(os.listdir(os.path.join(data_dir, 'gt')))
    with open(os.path.join(data_dir, 'errs.bin'), 'rb') as f:
        data = pickle.load(f)
        sd_errs, cos_errs = data['sd_errs'], data['cos_errs']
    for i, gt_name in enumerate(gt_names):
        verts, faces = load_ply(os.path.join(data_dir, 'gt', gt_name))
        image = render_mesh(verts.cuda(gpu_id), faces.cuda(gpu_id), R[0].cuda(gpu_id), t[0].cuda(gpu_id), 9)
        plt.imsave(os.path.join(data_dir, 'gt_imgs', '%06d.jpg' % i), image.cpu().numpy())

        if i < start_i:
            continue

        sd_err = sd_errs[i - start_i][0]
        assert sd_err.shape == (verts.shape[0],)
        max_dst = 0.1
        sd_err_nc = (sd_err / max_dst).clip(min=-1, max=1)
        colors = torch.zeros((verts.shape[0], 3))
        colors[sd_err_nc < 0] = (1 - sd_err_nc[sd_err_nc < 0].abs())[:, None] * torch.tensor([1, 1, 1])[None] + \
                                sd_err_nc[sd_err_nc < 0].abs()[:, None]  * torch.tensor([1, 0, 0])[None]
        colors[sd_err_nc >= 0] = (1 - sd_err_nc[sd_err_nc >= 0])[:, None] * torch.tensor([1, 1, 1])[None] + \
                                 sd_err_nc[sd_err_nc >= 0][:, None]  * torch.tensor([0, 1, 1])[None]
        image = render_mesh(verts.cuda(gpu_id), faces.cuda(gpu_id), R[0].cuda(gpu_id), t[0].cuda(gpu_id), 9, colors=colors.cuda(gpu_id))
        plt.imsave(os.path.join(data_dir, 'errs_imgs', '%06d.jpg' % i), image.cpu().numpy())
