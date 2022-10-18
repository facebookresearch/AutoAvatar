# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle

import utils.CAPE as cape_utils
from utils.render import *
from models.nets import *


class DynNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = copy.deepcopy(args)

        self.smpl_model = cape_utils.load_smpl(args)
        self.register_buffer('v_template', self.smpl_model.v_template, persistent=False)
        self.register_buffer('faces', self.smpl_model.faces, persistent=False)
        mask_ids = ['left_wrist', 'right_wrist', 'left_hand', 'right_hand', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot', 'head']
        mask_ids = [cape_utils.SMPL_JOINT_NAMES.index(e) for e in mask_ids]
        head_hands_feet_mask = self.smpl_model.lbs_weights[:, mask_ids].sum(dim=-1)     # (N,)
        head_hands_feet_mask[head_hands_feet_mask < 2e-2] = 0
        head_hands_feet_mask = (head_hands_feet_mask * 10).clip(max=1)
        self.register_buffer('head_hands_feet_mask', head_hands_feet_mask, persistent=False)
        W = cape_utils.compute_adjacent_matrix(self.smpl_model.parents, 4)
        self.register_buffer('W', W, persistent=False)      # (J + 1, J)

        data = np.load(args['data']['uv_info'])
        verts_uv, faces_uv, v2uv = torch.from_numpy(data['verts_uv']), torch.from_numpy(data['faces_uv']).long(), torch.from_numpy(data['v2uv']).long()
        self.geo_fn = UVRender(args, verts_uv, faces_uv, v2uv)
        self.register_buffer('head_hands_feet_mask_uv', self.geo_fn.to_uv(head_hands_feet_mask[None, :, None].cuda()), persistent=False)

        data = np.load(args['data']['resample_idxs_path'])
        self.resample_idxs = data['idxs']

        self.shape_enc_dec = ShapeEncDec(args)
        if args['model']['stage'] == 'auto_regr':
            self.dynamics_net = DynamicsNet(args)

    def compute_poses_feat(self, poses):
        """
        :param poses: (B, 69)
        """
        B = poses.shape[0]
        J = self.smpl_model.NUM_BODY_JOINTS
        N = self.smpl_model.get_num_verts()
        assert poses.shape == (B, 69)
        poses_quat = self.smpl_model.compute_poses_quat(poses)     # (B, J, 4)
        assert poses_quat.shape == (B, J, 4)
        lbs_w = self.smpl_model.lbs_weights[None].expand(B, N, J + 1)
        lbs_w = torch.einsum('bvj,jl->bvl', lbs_w, self.W)
        assert lbs_w.shape == (B, N, J)
        poses_feat = poses_quat[:, None] * lbs_w[..., None]
        assert poses_feat.shape == (B, N, J, 4)
        return poses_feat

    def normalize_sd_delta(self, sd_delta):
        sd_delta_nc = torch.sign(sd_delta) * (sd_delta.abs() * 1000 + 1).log() * 0.25
        return sd_delta_nc

    def normalize_globalRt(self, pts, poses):
        """
        :param pts: (B, M, 3)
        :param poses: (B, 75)
        """
        B, M, _ = pts.shape
        assert poses.shape == (B, 75)
        smpl_out = self.smpl_model(poses)
        root_T_inv = torch.linalg.inv(smpl_out.A[:, 0])     # (B, 4, 4)
        pts_nc = pts - poses[:, None, :3]
        pts_nc_homo = torch.ones((B, M, 1), dtype=torch.float, device=pts.device)
        pts_nc_homo = torch.cat([pts_nc, pts_nc_homo], dim=-1)
        pts_nc = torch.bmm(root_T_inv, pts_nc_homo.transpose(-2, -1)).transpose(-2, -1)[..., :3].contiguous()
        assert pts_nc.shape == (B, M, 3)
        return pts_nc

    def query_sdf_nets(self, pts, poses, shapes, force_coarse=False):
        """
        :param pts: (B, M, 3)
        :param poses: (B, 75)
        :param shapes: (B, N, C)
        """
        B, M, _ = pts.shape
        _, N, C = shapes.shape
        assert poses.shape == (B, 75) and shapes.shape == (B, N, C) and N == self.smpl_model.get_num_verts()
        verts = self.smpl_model(poses).vertices
        assert verts.shape == (B, N, 3)

        # Normalize global Rt
        verts = self.normalize_globalRt(verts, poses)
        pts = self.normalize_globalRt(pts, poses)

        # MLP decode
        # SMPL resample
        meshes = Meshes(verts=verts, faces=self.faces[None].expand(B, -1, -1))
        normals = meshes.verts_normals_padded()
        assert normals.shape == (B, N, 3)
        verts_ori = verts.clone()
        shapes_ori = shapes.clone()
        verts = verts[:, self.resample_idxs]
        normals = normals[:, self.resample_idxs]
        shapes = shapes[:, self.resample_idxs]
        N_ = verts.shape[1]
        assert verts.shape == (B, N_, 3) and normals.shape == (B, N_, 3) and shapes.shape == (B, N_, C)
        # KNN
        K = 20
        C_s = 64
        C_ = 128
        _, idx, pts_nn = knn_points(pts, verts, K=K, return_nn=True)
        assert torch.allclose(pts_nn, knn_gather(verts, idx))
        normals_nn = knn_gather(normals, idx)
        shapes_nn = knn_gather(shapes, idx)
        assert pts_nn.shape == (B, M, K, 3) and normals_nn.shape == (B, M, K, 3) and shapes_nn.shape == (B, M, K, C)
        pts_nn = pts_nn - pts[:, :, None]
        # Proj pts to mesh
        _, pts_proj, _, _, shapes_proj = cape_utils.proj_pts_to_mesh(pts, verts_ori, self.faces[None].expand(B, -1, -1).contiguous(), shapes_ori)
        assert pts_proj.shape == (B, M, 3) and shapes_proj.shape == (B, M, C)
        pts_proj = pts_proj - pts
        # Aggregate
        feat_nn = self.shape_enc_dec.pts_mlp(
            torch.cat([
                self.shape_enc_dec.pts_emb(pts_nn.view(B * M * K, 3)),
                self.shape_enc_dec.pts_emb(normals_nn.view(B * M * K, 3)),
                shapes_nn.view(B * M * K, C)[:, :C_s]
            ], dim=-1)
        ).view(B, M, K, C_)
        feat_proj = self.shape_enc_dec.proj_pts_mlp(
            torch.cat([
                self.shape_enc_dec.pts_emb(pts_proj.view(B * M, 3)),
                shapes_proj.view(B * M, C)[:, :C_s]
            ], dim=-1)
        ).view(B, M, 1, C_)
        feat = torch.cat([feat_nn, feat_proj], dim=-2)
        assert feat.shape == (B, M, K + 1, C_)
        w = self.shape_enc_dec.weights_fc(feat.view(B * M * (K + 1), C_)).view(B, M, K + 1, 1)
        w = torch.softmax(w, dim=-2)
        feat = (feat * w).sum(dim=-2)
        assert feat.shape == (B, M, C_)
        sdf = self.shape_enc_dec.sdf_mlp(feat.view(B * M, C_)).view(B, M)
        return sdf

    def compute_obpts(self, poses):
        """
        :param poses: (B, 75)
        """
        B = poses.shape[0]
        N = self.smpl_model.get_num_verts()
        K = self.args['model']['ob_vals'][-1]
        verts_smpl = self.smpl_model(poses).vertices
        meshes = Meshes(verts=verts_smpl, faces=self.faces[None].expand(B, -1, -1))
        normals_smpl = meshes.verts_normals_padded()    # (B, N, 3)
        offset = torch.linspace(*self.args['model']['ob_vals'], device=poses.device)[None, None, :, None] * normals_smpl[:, :, None, :]      # (B, N, K, 3)
        obpts = offset + verts_smpl[:, :, None]
        return obpts

    def shapes_to_obsdf(self, shapes, poses, mode='nets', faces=None):
        """
        :param shapes: (B, N, C)
        :param poses: (B, 75)
        """
        B = poses.shape[0]
        N = self.smpl_model.get_num_verts()
        K = self.args['model']['ob_vals'][-1]
        C = shapes.shape[-1]
        assert poses.shape == (B, 75) and shapes.shape[0] == B

        # Compute observer pts
        obpts = self.compute_obpts(poses)
        assert obpts.shape == (B, N, K, 3)

        # Query sdf
        if mode == 'meshes':
            assert C == 3 and faces is not None
            sdf, _, _, _, _ = cape_utils.proj_pts_to_mesh(obpts.view(B, N * K, 3), shapes, faces)
            sdf = sdf.view(B, N, K)
        elif mode == 'nets':
            assert shapes.shape == (B, N, C)
            sdf = self.query_sdf_nets(obpts.view(B, N * K, 3), poses, shapes, force_coarse=True)
            sdf = sdf.view(B, N, K)

        return sdf, obpts

    def query_sdf_with_grad(self, pts, poses, shapes):
        B, M, _ = pts.shape
        C = shapes.shape[-1]
        N = self.smpl_model.get_num_verts()
        assert pts.shape == (B, M, 3) and poses.shape == (B, 75) and shapes.shape == (B, N, C)

        with torch.enable_grad():
            pts.requires_grad_(True)
            sdf = self.query_sdf_nets(pts, poses, shapes)
            assert sdf.shape == (B, M)
            sdf_grad = autograd.grad([sdf.sum()], [pts], retain_graph=True, create_graph=True)[0]
            assert sdf_grad.shape == (B, M, 3)

        return sdf, sdf_grad

    def enc_shapes_to_sdf(self, obsdf, poses):
        """
        :param obsdf: (B, T, N, K)
        :param poses: (B, T, 75)
        """
        B, T, _ = poses.shape
        N = self.smpl_model.get_num_verts()
        K = self.args['model']['ob_vals'][-1]
        H = W = self.args['model']['uv_size']
        assert obsdf.shape == (B, T, N, K) and poses.shape == (B, T, 75)

        # Compute obpts_uv
        obpts = self.compute_obpts(poses.view(B * T, 75))
        assert obpts.shape == (B * T, N, K, 3)
        obpts = self.normalize_globalRt(obpts.view(B * T, N * K, 3), poses.view(B * T, 75))
        obpts_uv = self.geo_fn.to_uv(obpts.view(B * T, N, K * 3))
        assert obpts_uv.shape == (B * T, K * 3, H, W)

        # Compute obsdf_uv
        obsdf_uv = self.geo_fn.to_uv(obsdf.view(B * T, N, K))
        assert obsdf_uv.shape == (B * T, K, H, W)

        # Net forward
        in_feat = torch.cat([obpts_uv, obsdf_uv * 20], dim=1)
        shapes_uv = self.shape_enc_dec.shape_enc(in_feat)
        C = shapes_uv.shape[1]
        feat_uv_ = shapes_uv * (1 - self.head_hands_feet_mask_uv) + self.shape_enc_dec.uv_bias * self.head_hands_feet_mask_uv
        shapes = self.geo_fn.from_uv(feat_uv_)
        assert shapes.shape == (B * T, N, C) and shapes_uv.shape == (B * T, C, H, W)
        shapes = shapes.view(B, T, N, C)
        shapes_uv = shapes_uv.view(B, T, C, H, W)
        return shapes, shapes_uv

    def forward(self, obsdf, poses):
        """
        :param obsdf: (B, T, N, K)
        :param poses: (B, T_, 75)
        """
        N = self.smpl_model.get_num_verts()
        K = self.args['model']['ob_vals'][-1]
        H = W = self.args['model']['uv_size']

        if self.args['model']['stage'] == 'shape_enc_dec':
            B, T, _ = poses.shape
            assert obsdf.shape == (B, T, N, K) and poses.shape == (B, T, 75) and T == 1
            shapes, shapes_uv = self.enc_shapes_to_sdf(obsdf, poses)
            shapes = shapes.squeeze(1)
            C = shapes.shape[-1]
            assert shapes.shape == (B, N, C)

        elif self.args['model']['stage'] == 'auto_regr':
            B, T_, _ = poses.shape
            T = obsdf.shape[1]
            J_ = self.smpl_model.NUM_BODY_JOINTS + 1
            J = self.smpl_model.NUM_BODY_JOINTS
            assert obsdf.shape == (B, T, N, K) and poses.shape == (B, T_, 75) and T_ - T == 1 and T == self.args['model']['n_hist_frames']
            poses_ref = poses[:, -1:].expand(B, T_, 75).contiguous()

            # Compute obpts_uv
            obpts = self.compute_obpts(poses.view(B * T_, 75))
            assert obpts.shape == (B * T_, N, K, 3)
            obpts = self.normalize_globalRt(obpts.view(B * T_, N * K, 3), poses_ref.view(B * T_, 75))
            obpts_uv = self.geo_fn.to_uv(obpts.view(B * T_, N, K * 3))
            assert obpts_uv.shape == (B * T_, K * 3, H, W)
            obpts_uv = obpts_uv.view(B, T_ * K * 3, H, W)

            # Compute poses velocity
            poses_prev = poses[:, :-1].clone()
            poses_last = poses[:, 1:].clone()
            poses_vel = torch.zeros_like(poses_last)
            assert poses_prev.shape == (B, T, 75) and poses_last.shape == (B, T, 75) and poses_vel.shape == (B, T, 75)
            poses_vel[..., :3] = poses_last[..., :3] - poses_prev[..., :3]
            rot_prev = axis_angle_to_matrix(poses_prev[..., 3:].reshape(B * T * J_, 3))
            rot_last = axis_angle_to_matrix(poses_last[..., 3:].reshape(B * T * J_, 3))
            rot_vel = torch.bmm(rot_last, torch.linalg.inv(rot_prev))
            assert rot_vel.shape == (B * T * J_, 3, 3)
            poses_vel[..., 3:] = matrix_to_axis_angle(rot_vel).view(B, T, J_ * 3)

            poses_vel_feat = self.compute_poses_feat(poses_vel[..., 6:].reshape(B * T, 69))
            assert poses_vel_feat.shape == (B * T, N, J, 4)
            poses_vel_feat = torch.cat([poses_vel_feat.view(B * T, N, J * 4), poses_vel[..., :6].reshape(B * T, 1, 6).expand(B * T, N, 6)], dim=-1)
            assert poses_vel_feat.shape == (B * T, N, J * 4 + 6)
            poses_vel_feat_uv = self.geo_fn.to_uv(poses_vel_feat)
            assert poses_vel_feat_uv.shape == (B * T, J * 4 + 6, H, W)
            poses_vel_feat_uv = self.dynamics_net.local_poses_vel_conv_block(poses_vel_feat_uv).view(B, T * 32, H, W)
            poses_vel_feat_uv = self.dynamics_net.temp_poses_vel_conv_block(poses_vel_feat_uv)
            assert poses_vel_feat_uv.shape == (B, 32, H, W)

            # Compute pose_feat
            pose_feat = self.compute_poses_feat(poses[:, -1, 6:].clone())
            assert pose_feat.shape == (B, N, J, 4)
            pose_feat_uv = self.geo_fn.to_uv(pose_feat.view(B, N, J * 4))
            assert pose_feat_uv.shape == (B, J * 4, H, W)
            pose_feat_uv = self.dynamics_net.local_pose_conv_block(pose_feat_uv)
            assert pose_feat_uv.shape == (B, 32, H, W)

            # Compute obsdf_feat_uv
            obsdf_delta = obsdf[:, 1:] - obsdf[:, :-1]
            assert obsdf_delta.shape == (B, T - 1, N, K)
            obsdf_delta = self.normalize_sd_delta(obsdf_delta)
            obsdf_delta = obsdf_delta.permute(0, 2, 1, 3).contiguous()
            assert obsdf_delta.shape == (B, N, T - 1, K)
            obsdf_feat = torch.cat([obsdf_delta.view(B, N, (T - 1) * K), obsdf[:, -1] * 20], dim=-1)
            assert obsdf_feat.shape == (B, N, T * K)
            obsdf_feat_uv = self.geo_fn.to_uv(obsdf_feat)
            assert obsdf_feat_uv.shape == (B, T * K, H, W)

            # Unet forward
            feat_uv = torch.cat([obpts_uv, poses_vel_feat_uv, pose_feat_uv, obsdf_feat_uv], dim=1)
            shapes_uv_delta = self.dynamics_net.unet(feat_uv)
            _, shapes_uv_prev = self.enc_shapes_to_sdf(obsdf[:, -1:], poses[:, -2:-1])
            shapes_uv = shapes_uv_prev[:, 0] + shapes_uv_delta
            C = shapes_uv.shape[1]
            feat_uv_ = shapes_uv * (1 - self.head_hands_feet_mask_uv) + self.shape_enc_dec.uv_bias * self.head_hands_feet_mask_uv
            shapes = self.geo_fn.from_uv(feat_uv_)
            assert shapes.shape == (B, N, C)

        return shapes


class ShapeEncDec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = copy.deepcopy(args)

        self.sdf_mlp = MLP([128, 128, 1], [-1], 'softplus', True, 'linear', False)
        self.pts_emb = Embedder(3, 4)
        self.proj_pts_mlp = MLP([64 + self.pts_emb.out_ch, 128, 128], [-1], 'softplus', True, 'softplus', True)
        self.pts_mlp = MLP([64 + self.pts_emb.out_ch * 2, 128, 128], [-1], 'softplus', True, 'softplus', True)
        self.weights_fc = nn.Linear(128, 1)
        self.shape_enc = ShapeEnc(args)
        self.register_parameter('uv_bias', nn.Parameter(torch.normal(0, 0.01, (1, 64, 256, 256), dtype=torch.float), requires_grad=True))


class DynamicsNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = copy.deepcopy(args)

        self.local_pose_conv_block = ConvBlock(92, 32, args['model']['uv_size'], kernel_size=1, padding=0)
        self.local_poses_vel_conv_block = ConvBlock(98, 32, args['model']['uv_size'], kernel_size=1, padding=0)
        self.temp_poses_vel_conv_block = ConvBlock(32 * args['model']['n_hist_frames'], 32, args['model']['uv_size'], kernel_size=1, padding=0)
        self.unet = Unet(args)


class ShapeEnc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = copy.deepcopy(args)

        self.conv_in = ConvBlock(args['model']['ob_vals'][-1] * 4, 64, 256)
        self.conv0 = ConvDownBlock(64, 64, 256)
        self.conv1 = ConvDownBlock(64, 64, 128)
        self.conv2 = ConvUpBlock(64, 64, 128)
        self.conv3 = ConvUpBlock(64, 64, 256)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_in(x)
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1) + x0
        x3 = self.conv3(x2) + x
        out = self.conv_out(x3)
        return out


class Unet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = copy.deepcopy(args)

        self.conv_in = ConvBlock((args['model']['n_hist_frames'] + args['model']['n_batch_frames'] * 3) * args['model']['ob_vals'][-1] + 64, 64, 256)

        self.conv_down0 = ConvDownBlock(64, 128, 256)
        self.conv_down1 = ConvDownBlock(128, 256, 128)
        self.conv_down2 = ConvDownBlock(256, 256, 64)
        self.conv_down3 = ConvDownBlock(256, 256, 32)

        self.conv_up3 = ConvUpBlock(256, 256, 32)
        self.conv_up2 = ConvUpBlock(256, 256, 64)
        self.conv_up1 = ConvUpBlock(256, 128, 128)
        self.conv_up0 = ConvUpBlock(128, 64, 256)

        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.Tanh()
        )
        torch.nn.init.zeros_(self.conv_out[0].weight)
        if hasattr(self.conv_out[0], 'bias') and self.conv_out[0].bias is not None:
            torch.nn.init.zeros_(self.conv_out[0].bias)

    def forward(self, x):
        x = self.conv_in(x)

        x0 = self.conv_down0(x)
        x1 = self.conv_down1(x0)
        x2 = self.conv_down2(x1)
        x3 = self.conv_down3(x2)

        y3 = self.conv_up3(x3) + x2
        y2 = self.conv_up2(y3) + x1
        y1 = self.conv_up1(y2) + x0
        y0 = self.conv_up0(y1) + x

        out = self.conv_out(y0)
        return out
