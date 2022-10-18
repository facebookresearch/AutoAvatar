# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle
import numpy as np
import os
import copy
import shutil, inspect
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch3d.io import save_ply
import time

import utils.CAPE as cape_utils
from utils.implicit import reconstruction
from models.PosedDecKNN_dPoses_dHs.nets import DynNet
import models.std.visual as visual


class Implicit_Trainbox(pl.LightningModule):
    def __init__(self, args, log_dir, resolution, recurrent=True, eval_frames=None, pose_model=None):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.log_dir = log_dir
        self.resolution = resolution
        self.recurrent = recurrent
        self.eval_frames = eval_frames
        self.pose_model = pose_model

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(os.path.join(log_dir, 'ckpt')):
            os.mkdir(os.path.join(log_dir, 'ckpt'))
        if not os.path.exists(os.path.join(log_dir, 'net_def')):
            os.mkdir(os.path.join(log_dir, 'net_def'))
        if not os.path.exists(os.path.join(log_dir, 'mesh')):
            os.mkdir(os.path.join(log_dir, 'mesh'))

        shutil.copy(os.path.realpath(__file__), os.path.join(log_dir, 'net_def'))
        shutil.copy(inspect.getfile(DynNet), os.path.join(log_dir, 'net_def'))

        self.dyn_net = DynNet(args, eval_frames)
        self.itr = 0

    def save_ckpt(self):
        torch.save(self.dyn_net.state_dict(), os.path.join(self.log_dir, 'ckpt', 'dyn_net_%06d.pth' % self.itr))

    def load_ckpt(self, itr, log_dir):
        self.dyn_net.load_state_dict(torch.load(os.path.join(log_dir, 'ckpt', 'dyn_net_%06d.pth' % itr), map_location='cpu'))

    def preprocess(self, batch):
        verts_detail, faces_detail, verts_smt, faces_smt, poses = batch['verts_detail'], batch['faces_detail'], batch['verts_smt'], batch['faces_smt'], batch['poses']
        B, T, _ = poses.shape
        N = self.dyn_net.smpl_model.v_template.shape[0]
        verts_smpl = self.dyn_net.smpl_model(poses.view(B * T, 75)).vertices.view(B, T, N, 3)
        if len(verts_detail) == 0:
            verts_detail = [verts_smpl[:, i].contiguous() for i in range(T)]
            faces_detail = [self.dyn_net.smpl_model.faces.to(verts_smpl.device)[None].expand(B, -1, -1).contiguous() for i in range(T)]
            verts_smt = verts_detail
            faces_smt = faces_detail
        if not self.args['data']['separate_detail']:
            verts_smt = verts_detail
            faces_smt = faces_detail
        return verts_detail, faces_detail, verts_smt, faces_smt, poses, verts_smpl

    def train_or_valid_step(self, batch, batch_idx, is_train):
        verts_detail_all, faces_detail_all, verts_smt_all, faces_smt_all, poses_all, verts_smpl_all = self.preprocess(batch)

        T = self.args['model']['n_batch_frames']
        T_hist = self.args['model']['n_hist_frames']
        T_next = T - T_hist
        K = self.args['model']['ob_vals'][-1]
        assert T_next == 1

        iter_times = []
        sd_errs = []
        cos_errs = []
        obsdf_rollout = None
        shapes_uv_init = None
        loss_surf_sdf = 0
        loss_surf_grad = 0
        loss_igr = 0
        loss_o = 0
        end_idx = self.args['train']['n_rollout']
        if self.eval_frames is not None and batch_idx + T_hist == self.eval_frames[0]:
            end_idx = poses_all.shape[1] - T + 1
        for i in range(end_idx):
            verts_detail = verts_detail_all[i:i+T]
            faces_detail = faces_detail_all[i:i+T]
            verts_smt = verts_smt_all[i:i+T]
            faces_smt = faces_smt_all[i:i+T]
            poses = poses_all[:, i:i+T]
            verts_smpl = verts_smpl_all[:, i:i+T]
            N = verts_smpl.shape[2]
            B = poses.shape[0]
            if poses.shape[1] < T:
                break

            verts = verts_smt
            faces = faces_smt
            verts_gt = verts_smt if not self.args['model']['use_detail'] else verts_detail
            faces_gt = faces_smt if not self.args['model']['use_detail'] else faces_detail
            bbmin = verts_smpl.min(dim=2)[0] - 0.1
            bbmax = verts_smpl.max(dim=2)[0] + 0.1
            surf_pts, surf_normals, igr_pts, bbox_pts, rand_pts = cape_utils.sample_igr_pts(verts_gt[-1], faces_gt[-1], bbmin[:, -1], bbmax[:, -1], self.args)

            # start_time = time.time()

            if self.args['model']['stage'] == 'shape_enc_dec':
                obsdf, _ = self.dyn_net.shapes_to_obsdf(verts[-1], poses[:, -1], mode='meshes', faces=faces[-1])
                assert obsdf.shape == (B, N, 1)
                obsdf = obsdf[:, None]

            if self.args['model']['stage'] == 'auto_regr':
                if obsdf_rollout is None:
                    if 'verts_init' not in batch:
                        obsdf = [self.dyn_net.shapes_to_obsdf(verts[j], poses[:, j], mode='meshes', faces=faces[j])[0] for j in range(T_hist)]
                        obsdf = torch.stack(obsdf, dim=1)
                        assert obsdf.shape == (B, T_hist, N, 1)
                    else:
                        _, shapes_uv_pose = self.pose_model(None, poses[:, :-1].reshape(B * T_hist, 1, 75))
                        obsdf, _ = self.pose_model.shapes_to_obsdf(torch.zeros((B * T_hist, 0, 0), device=poses.device), poses[:, :-1].reshape(B * T_hist, 75), mode='nets', shapes_uv=shapes_uv_pose)
                        assert obsdf.shape == (B * T_hist, N, 1)
                        obsdf = obsdf.view(B, T_hist, N, 1).contiguous()
                        # obsdf = self.dyn_net.shapes_to_obsdf(batch['verts_init'], batch['poses_init'], mode='meshes', faces=batch['faces_init'])[0]
                        # assert obsdf.shape == (B, N, 1)
                        # obsdf = obsdf[:, None].expand(B, T_hist, N, 1).contiguous()
                else:
                    obsdf = obsdf_rollout.detach()

            shapes, shapes_uv = self.dyn_net(obsdf, poses)
            if self.eval_frames is not None:
                if shapes_uv_init is None:
                    shapes_uv_init = shapes_uv
                else:
                    shapes_uv = shapes_uv * (1 - self.dyn_net.head_hands_feet_mask_uv) + shapes_uv_init * self.dyn_net.head_hands_feet_mask_uv

            if i + 1 < end_idx and self.recurrent:
                with torch.no_grad():
                    obsdf_new, _ = self.dyn_net.shapes_to_obsdf(shapes, poses[:, -1], mode='nets', shapes_uv=shapes_uv)
                    obsdf_rollout = torch.cat([obsdf[:, 1:], obsdf_new[:, None]], dim=1).detach()
                    assert obsdf_rollout.shape == (B, T_hist, N, 1)

            if self.eval_frames is None:
                # Losses
                surf_sdf, surf_sdf_grad = self.dyn_net.query_sdf_with_grad(surf_pts, poses[:, -1], shapes_uv)
                rand_sdf, rand_sdf_grad = self.dyn_net.query_sdf_with_grad(rand_pts, poses[:, -1], shapes_uv)
                bbox_sdf = rand_sdf[:, self.args['train']['n_pts_scan_igr']:]
                assert bbox_sdf.shape == (B, self.args['train']['n_pts_bbox_igr'])
                loss_surf_sdf += surf_sdf.abs().mean() / self.args['train']['n_rollout']
                loss_surf_grad += torch.norm(surf_sdf_grad - surf_normals, p=2, dim=-1).mean() / self.args['train']['n_rollout']
                loss_igr += (torch.norm(rand_sdf_grad, p=2, dim=-1) - 1).pow(2).mean() / self.args['train']['n_rollout']
                loss_o += torch.exp(-50.0 * torch.abs(bbox_sdf)).mean() / self.args['train']['n_rollout']
            else:
                out_dir = os.path.join(self.log_dir, 'mesh', 'batch_%06d' % batch_idx)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                if not os.path.exists(os.path.join(out_dir, 'gt')):
                    os.mkdir(os.path.join(out_dir, 'gt'))
                if not os.path.exists(os.path.join(out_dir, 'pred')):
                    os.mkdir(os.path.join(out_dir, 'pred'))
                if not os.path.exists(os.path.join(out_dir, 'poses')):
                    os.mkdir(os.path.join(out_dir, 'poses'))
                with open(os.path.join(out_dir, 'poses', 'poses_%06d.bin' % (i + T_hist)), 'wb') as f:
                    pickle.dump({'poses': poses.cpu()}, f)
                if i == 0:
                    for j in range(T_hist):
                        save_ply(os.path.join(out_dir, 'gt', 'gt_%06d.ply' % j), verts_gt[j][0], faces_gt[j][0])
                save_ply(os.path.join(out_dir, 'gt', 'gt_%06d.ply' % (i + T_hist)), verts_gt[-1][0], faces_gt[-1][0])
                out = reconstruction(self.dyn_net.query_sdf_nets, poses.device, None,
                                     self.resolution, bbmin[0, -1].cpu().numpy(), bbmax[0, -1].cpu().numpy(),
                                     use_octree=False, num_samples=4096, transform=None, thresh=0, texture_net = None, poses=poses[:1, -1], shapes=shapes_uv[:1])
                verts_out, faces_out = out
                verts_out, faces_out = torch.from_numpy(verts_out).float().to(poses.device), torch.from_numpy(faces_out.astype(np.int32)).long().to(poses.device)
                save_ply(os.path.join(out_dir, 'pred', 'pred_%06d.ply' % (i + T_hist)), verts_out, faces_out)
                sd_err, cos_err = cape_utils.scan_to_pred_errors(verts_gt[-1], faces_gt[-1], verts_out[None], faces_out[None])
                sd_errs.append(sd_err.cpu())
                cos_errs.append(cos_err.cpu())

        #     iter_time = time.time() - start_time
        #     iter_times.append(iter_time)
        #     print('time:', iter_time)
        # print('mean time:', np.array(iter_times[1:-1]).mean())
        # input('pause')

        if self.eval_frames is not None:
            with open(os.path.join(out_dir, 'errs.bin'), 'wb') as f:
                pickle.dump({'sd_errs': sd_errs, 'cos_errs': cos_errs}, f)

            visual.render_meshes(out_dir, start_i=T_hist, simplify_mesh=False)
            os.system('bash models/std/videos.sh %s %s' % (out_dir, str(T_hist)))

        loss = loss_surf_sdf + loss_surf_grad + loss_igr * self.args['train']['lambda_igr'] + loss_o * self.args['train']['lambda_o']

        res_dict = {
            'verts': verts,
            'faces': faces,
            'verts_gt': verts_gt,
            'faces_gt': faces_gt,
            'verts_smpl': verts_smpl,
            'poses': poses,
            'shapes': shapes_uv,
            'bbmin': bbmin,
            'bbmax': bbmax,

            'loss_surf_sdf': loss_surf_sdf,
            'loss_surf_grad': loss_surf_grad,
            'loss_igr': loss_igr,
            'loss_o': loss_o,
            'loss': loss
        }
        
        return res_dict

    def training_step(self, batch, batch_idx):
        res_dict = self.train_or_valid_step(batch, batch_idx, True)

        # log
        prefix = 'Train'
        self.log('%s/loss' % prefix, res_dict['loss'])
        self.log('%s/loss_surf_sdf' % prefix, res_dict['loss_surf_sdf'])
        self.log('%s/loss_surf_grad' % prefix, res_dict['loss_surf_grad'])
        self.log('%s/loss_igr' % prefix, res_dict['loss_igr'])
        self.log('%s/loss_o' % prefix, res_dict['loss_o'])

        # checkpoint
        self.itr += 1
        if self.itr % self.args['train']['ckpt_step'] == 0:
            self.save_ckpt()

        return res_dict['loss']

    def validation_step(self, batch, batch_idx):
        if self.eval_frames is not None and batch_idx + self.args['model']['n_hist_frames'] not in self.eval_frames:
            return
        res_dict = self.train_or_valid_step(batch, batch_idx, False)

        # log
        prefix = 'Valid'
        self.log('%s/loss' % prefix, res_dict['loss'])
        self.log('%s/loss_surf_sdf' % prefix, res_dict['loss_surf_sdf'])
        self.log('%s/loss_surf_grad' % prefix, res_dict['loss_surf_grad'])
        self.log('%s/loss_igr' % prefix, res_dict['loss_igr'])
        self.log('%s/loss_o' % prefix, res_dict['loss_o'])

        if self.eval_frames is None:
            self.compute_meshes(res_dict, batch, batch_idx)

    def configure_optimizers(self):
        if self.args['model']['stage'] == 'shape_enc_dec':
            optimizer = optim.Adam(self.dyn_net.parameters(), lr=self.args['train']['lr'])
        elif self.args['model']['stage'] == 'auto_regr':# and not self.args['model']['use_detail']:
            optimizer = optim.Adam(self.dyn_net.parameters(), lr=self.args['train']['lr'])
        # elif self.args['model']['use_detail']:
        #     optimizer = optim.Adam(self.dyn_net.detail_dec.parameters(), lr=self.args['train']['lr'])
        return optimizer

    def compute_meshes(self, res_dict, batch, batch_idx):
        verts, faces, verts_gt, faces_gt, verts_smpl, poses, shapes, bbmin, bbmax = res_dict['verts'], res_dict['faces'], res_dict['verts_gt'], res_dict['faces_gt'], \
            res_dict['verts_smpl'], res_dict['poses'], res_dict['shapes'], res_dict['bbmin'], res_dict['bbmax']

        T = self.args['model']['n_batch_frames']
        T_hist = self.args['model']['n_hist_frames']
        T_next = T - T_hist

        if not os.path.exists(os.path.join(self.log_dir, 'mesh', 'itr_%06d' % self.itr)):
            os.mkdir(os.path.join(self.log_dir, 'mesh', 'itr_%06d' % self.itr))
        out_dir = os.path.join(self.log_dir, 'mesh', 'itr_%06d' % self.itr, 'batch_%06d' % batch_idx)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for i in range(T_hist):
            save_ply(os.path.join(out_dir, 'hist_%d.ply' % i), verts[i][0], faces[i][0])
        save_ply(os.path.join(out_dir, 'gt.ply'), verts_gt[-1][0], faces_gt[-1][0])
        out = reconstruction(self.dyn_net.query_sdf_nets, poses.device, None,
                             self.resolution, bbmin[0, -1].cpu().numpy(), bbmax[0, -1].cpu().numpy(),
                             use_octree=False, num_samples=4096, transform=None, thresh=0, texture_net = None, poses=poses[:1, -1], shapes=shapes[:1])
        if out != -1:
            verts_out, faces_out = out
            save_ply(os.path.join(out_dir, 'pred.ply'),
                     torch.from_numpy(verts_out).float().contiguous(), torch.from_numpy(faces_out.astype(np.int32)).contiguous().long())

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
