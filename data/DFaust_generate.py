# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np
import h5py
import sys
import os
import copy
import pickle
import yaml
import smplx
import open3d as o3d
from tqdm import tqdm
from pytorch3d.io import save_ply, load_obj, load_ply
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
from pytorch3d.ops import knn_points

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

import utils.DFaust as dfaust_utils
import utils.CAPE as cape_utils
from utils.configs import *


def generate_DFaust_SMPLH(data_dir, smpl_dir, out_dir, subject, subject_gender, gpu_id=0):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, subject)):
        os.mkdir(os.path.join(out_dir, subject))

    bm_fname = os.path.join(smpl_dir, 'smplh/%s/model.npz' % subject_gender)
    dmpl_fname = os.path.join(smpl_dir, 'dmpls/%s/model.npz' % subject_gender)
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)#.cuda(gpu_id)
    faces = c2c(bm.f)

    npz_files = sorted(os.listdir(os.path.join(data_dir, 'DFaust_67', subject)))
    for npz_file in npz_files:
        if '_poses' not in npz_file:
            continue
        if not os.path.exists(os.path.join(out_dir, subject, npz_file[:-4])):
            os.mkdir(os.path.join(out_dir, subject, npz_file[:-4]))
        bdata = np.load(os.path.join(data_dir, 'DFaust_67', subject, npz_file))
        time_length = len(bdata['trans'])
        body_parms = {
            'root_orient': torch.Tensor(bdata['poses'][:, :3]),#.cuda(gpu_id), # controls the global root orientation
            'pose_body': torch.Tensor(bdata['poses'][:, 3:66]),#.cuda(gpu_id), # controls the body
            'pose_hand': torch.Tensor(bdata['poses'][:, 66:]),#.cuda(gpu_id), # controls the finger articulation
            'trans': torch.Tensor(bdata['trans']),#.cuda(gpu_id), # controls the global body position
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)),#.cuda(gpu_id), # controls the body shape. Body shape is static
            'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]),#.cuda(gpu_id) # controls soft tissue dynamics
        }
        body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'root_orient', 'trans']})
        for i in tqdm(range(time_length)):
            verts = c2c(body_pose_beta.v[i])
            verts = torch.from_numpy(verts)
            verts_ = verts.clone()
            verts[:, 1] = verts_[:, 2]
            verts[:, 2] = -verts_[:, 1]
            save_ply(os.path.join(out_dir, subject, npz_file[:-4], '%06d.ply' % i), verts, torch.from_numpy(faces))


def smplh_to_smpl(data_dir, subject, smpl_model_path, gpu_id=0):
    if not os.path.exists(os.path.join(data_dir, 'smpl_poses')):
        os.mkdir(os.path.join(data_dir, 'smpl_poses'))
    if not os.path.exists(os.path.join(data_dir, 'smpl_poses', subject)):
        os.mkdir(os.path.join(data_dir, 'smpl_poses', subject))

    with open('data/smplh2smpl.yaml', 'r') as f:
        default_configs = yaml.load(f, Loader=yaml.FullLoader)
    seqs = sorted(os.listdir(os.path.join(data_dir, 'smplh_meshes', subject)))
    for seq in seqs:
        if not os.path.exists(os.path.join(data_dir, 'smpl_poses', subject, seq)):
            os.mkdir(os.path.join(data_dir, 'smpl_poses', subject, seq))
        configs = copy.deepcopy(default_configs)
        configs['body_model']['folder'] = smpl_model_path
        configs['datasets']['mesh_folder']['data_folder'] = os.path.join(data_dir, 'smplh_meshes', subject, seq)
        configs['output_folder'] = os.path.join(data_dir, 'smpl_poses', subject, seq)
        with open('tmp/configs.yaml', 'w') as f:
            yaml.dump(configs, f)
        os.system('cd external/smplx | python -m transfer_model --exp-cfg tmp/configs.yaml')


def DFaust_smplh_to_smpl(dataset_dir, smpl_dir, out_dir, subject, subject_gender, gpu_id=0):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, subject)):
        os.mkdir(os.path.join(out_dir, subject))

    bm_fname = os.path.join(smpl_dir, 'smplh/%s/model.npz' % subject_gender)
    dmpl_fname = os.path.join(smpl_dir, 'dmpls/%s/model.npz' % subject_gender)
    num_betas = 10 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)#.cuda(gpu_id)
    faces = c2c(bm.f)

    npz_files = sorted(os.listdir(os.path.join(dataset_dir, 'DFaust_67', subject)))
    for npz_file in npz_files:
        if '_poses' not in npz_file:
            continue
        bdata = np.load(os.path.join(dataset_dir, 'DFaust_67', subject, npz_file))
        time_length = len(bdata['trans'])
        body_parms = {
            'root_orient': torch.Tensor(bdata['poses'][:, :3]),#.cuda(gpu_id), # controls the global root orientation
            'pose_body': torch.Tensor(bdata['poses'][:, 3:66]),#.cuda(gpu_id), # controls the body
            'pose_hand': torch.Tensor(bdata['poses'][:, 66:]),#.cuda(gpu_id), # controls the finger articulation
            'trans': torch.Tensor(bdata['trans']),#.cuda(gpu_id), # controls the global body position
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)),#.cuda(gpu_id), # controls the body shape. Body shape is static
            'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]),#.cuda(gpu_id) # controls soft tissue dynamics
        }
        body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'root_orient', 'trans']})

        root_joints = body_pose_beta.Jtr[:, 0]
        smpl_poses = torch.Tensor(bdata['poses'][:, 3:72])
        global_orient = torch.Tensor(bdata['poses'][:, :3])
        transls = torch.Tensor(bdata['trans'])
        flip_yz_mat = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).float()[None].expand(time_length, 3, 3)

        global_rotmat = axis_angle_to_matrix(global_orient)
        global_rotmat = torch.bmm(flip_yz_mat, global_rotmat)
        global_orient = matrix_to_axis_angle(global_rotmat)

        root_joints_yup = torch.bmm(flip_yz_mat, root_joints[..., None])[..., 0]
        root_joints_notransl = root_joints - transls
        transls = root_joints_yup - root_joints_notransl

        poses = torch.cat([transls, global_orient, smpl_poses], dim=-1)
        assert poses.shape == (time_length, 75)
        np.savez_compressed(os.path.join(out_dir, subject, npz_file), poses=poses.numpy())

    # Save template
    shape_data = np.load(os.path.join(dataset_dir, 'DFaust_67', subject, 'shape.npz'))
    betas = torch.Tensor(shape_data['betas'][:10]).unsqueeze(0)
    body_pose_beta = bm(betas=betas)
    verts = c2c(body_pose_beta.v[0])
    verts = torch.from_numpy(verts)
    # verts_ = verts.clone()
    # verts[:, 1] = verts_[:, 2]
    # verts[:, 2] = -verts_[:, 1]
    save_ply(os.path.join(out_dir, subject, 'v_template.ply'), verts, torch.from_numpy(faces))


def DFaust_parse_raw(dataset_dir, subject):
    dfaust_json = dfaust_utils.DFaustJson()
    seq_names = sorted(os.listdir(os.path.join(dataset_dir, 'scans', subject)))
    seqs = []
    for seq_name in seq_names:
        ply_names = sorted(os.listdir(os.path.join(dataset_dir, 'scans', subject, seq_name)))
        poses = np.load(os.path.join(dataset_dir, 'smpl_poses', subject, '%s_%s_poses.npz' % (subject, seq_name)))['poses']
        pre_idx = None
        frames = []
        for i, ply_name in enumerate(ply_names):
            idx = int(ply_name.split('.')[-2])
            if pre_idx is not None and idx != pre_idx + 1:
                seqs = dfaust_json.append_seqs(seqs, seq_name, frames)
                frames = []
            frames = dfaust_json.append_frames(frames, os.path.join('scans', subject, seq_name, ply_name), poses[i])
            pre_idx = idx
        seqs = dfaust_json.append_seqs(seqs, seq_name, frames)
    dfaust_json.set_data(subject, seqs)
    dfaust_json.dump_bin_file(os.path.join(dataset_dir, '%s_raw.bin' % subject))

    print(dfaust_json.num_of_seqs())
    print(dfaust_json.num_of_frames())
    for seq in dfaust_json.data['seqs']:
        print(seq['id'], seq['seq_name'])


def split_train_test(dataset_dir, tag, bin_path, subject, interp_acts, extrap_acts):
    dfaust_json = dfaust_utils.DFaustJson(bin_path)
    dfaust_json_new = dfaust_utils.DFaustJson()
    seqs_new = []
    train_list = []
    interp_list = []
    extrap_list = []
    for seq in dfaust_json.data['seqs']:
        if seq['id'] in extrap_acts[0]:
            assert seq['seq_name'] in extrap_acts[1]
            seqs_new = dfaust_json_new.append_seqs(seqs_new, seq['seq_name'], copy.deepcopy(seq['frames']))
            extrap_list.append(seqs_new[-1]['id'])
        elif seq['id'] in interp_acts[0]:
            assert seq['seq_name'] in interp_acts[1]
            half_len = len(seq['frames']) // 2
            seqs_new = dfaust_json_new.append_seqs(seqs_new, seq['seq_name'], copy.deepcopy(seq['frames'][:half_len]))
            train_list.append(seqs_new[-1]['id'])
            seqs_new = dfaust_json_new.append_seqs(seqs_new, seq['seq_name'], copy.deepcopy(seq['frames'][half_len:]))
            interp_list.append(seqs_new[-1]['id'])
        else:
            seqs_new = dfaust_json_new.append_seqs(seqs_new, seq['seq_name'], copy.deepcopy(seq['frames']))
            train_list.append(seqs_new[-1]['id'])

    dfaust_json_new.set_data(subject, seqs_new)
    dfaust_json_new.dump_bin_file(os.path.join(dataset_dir, '%s_%s.bin' % (subject, tag)))
    print(dfaust_json_new.num_of_seqs())
    print(dfaust_json_new.num_of_frames())

    with open(os.path.join(dataset_dir, '%s_%s_train.bin' % (subject, tag)), 'wb') as f:
        pickle.dump(train_list, f)
    with open(os.path.join(dataset_dir, '%s_%s_interp.bin' % (subject, tag)), 'wb') as f:
        pickle.dump(interp_list, f)
    with open(os.path.join(dataset_dir, '%s_%s_extrap.bin' % (subject, tag)), 'wb') as f:
        pickle.dump(extrap_list, f)
    print(train_list)
    print(interp_list)
    print(extrap_list)


def add_transl(dataset_dir, bin_path, subject, smpl_path):
    smpl_model = smplx.SMPLLayer(model_path=smpl_path)
    dfaust_json = dfaust_utils.DFaustJson(bin_path)
    betas = []
    for seq in tqdm(dfaust_json.data['seqs']):
        bdata = np.load(os.path.join(dataset_dir, 'DFaust_67', subject, '%s_%s_poses.npz' % (subject, seq['seq_name'])))
        for i in range(len(seq['frames'])):
            frame = seq['frames'][i]
            idx = int(frame['pose_path'].split('/')[-1][:-4])
            with open(os.path.join(dataset_dir, frame['pose_path']), 'rb') as f:
                data = pickle.load(f)
            verts_smpl_ref, _, _ = load_obj(os.path.join(dataset_dir, frame['pose_path'][:-4] + '.obj'))
            body_pose = data['full_pose'][0].detach().cpu()[None, 1:]
            global_orient = data['full_pose'][0].detach().cpu()[None, 0]
            verts_smpl = smpl_model(betas=data['betas'].detach().cpu(), body_pose=body_pose, global_orient=global_orient).vertices[0]
            transl = (verts_smpl_ref - verts_smpl).mean(dim=0)
            rot = matrix_to_axis_angle(data['full_pose'][0].detach().cpu())
            assert rot.shape == (24, 3)
            poses = np.concatenate([transl, rot.view(72).numpy()], axis=0)
            assert poses.shape == (75,)
            frame['poses'] = poses
            betas.append(data['betas'].detach().cpu())
    betas = torch.cat(betas, dim=0).mean(dim=0)[None]
    v_template = smpl_model(betas=betas).vertices[0]
    save_ply(os.path.join(dataset_dir, 'smpl_poses', subject, 'v_template.ply'), v_template, smpl_model.faces_tensor)
    dfaust_json.dump_bin_file(bin_path)


def simplify_scans(ws_dir, dataset_dir, bin_path, config_path):
    dfaust_json = dfaust_utils.DFaustJson(bin_path)
    smpl_model = cape_utils.load_smpl(load_configs(config_path, ws_dir)).cuda()
    if not os.path.exists(os.path.join(dataset_dir, 'scans_simple')):
        os.mkdir(os.path.join(dataset_dir, 'scans_simple'))
    if not os.path.exists(os.path.join(dataset_dir, 'scans_simple', dfaust_json.data['subject'])):
        os.mkdir(os.path.join(dataset_dir, 'scans_simple', dfaust_json.data['subject']))

    for seq in tqdm(dfaust_json.data['seqs']):
        mesh_dir = os.path.join(dataset_dir, 'scans_simple', dfaust_json.data['subject'], seq['seq_name'])
        if not os.path.exists(mesh_dir):
            os.mkdir(mesh_dir)
        for i in tqdm(range(len(seq['frames']))):
            frame = seq['frames'][i]
            ply_path = os.path.join(dataset_dir, frame['ply_path'])
            verts, faces = load_ply(ply_path)
            verts, faces = verts.cuda(), faces.cuda()
            poses = torch.from_numpy(frame['poses']).float().cuda()[None]
            verts_smpl = smpl_model(poses).vertices[0]
            bbmin = verts_smpl.min(dim=0)[0][None] - 0.1
            bbmax = verts_smpl.max(dim=0)[0][None] + 0.1
            mask_min = (verts > bbmin).long().cumprod(dim=-1)[:, -1]
            mask_max = (verts < bbmax).long().cumprod(dim=-1)[:, -1]
            verts_mask = mask_min * mask_max
            faces_mask = verts_mask[faces[:, 0]] * verts_mask[faces[:, 1]] * verts_mask[faces[:, 2]]
            faces_val = faces[faces_mask.bool()]
            verts_idxs_new2old = torch.arange(0, verts.shape[0]).long()[verts_mask.bool()]
            verts_idxs_old2new = torch.zeros_like(verts_mask) - 1
            verts_idxs_old2new[verts_idxs_new2old] = torch.arange(0, verts_idxs_new2old.shape[0]).long().cuda()
            faces = verts_idxs_old2new[faces_val]
            verts = verts[verts_idxs_new2old]
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(int(faces.shape[0] * 0.075 * 0.67))
            verts, faces = torch.from_numpy(np.asarray(mesh_o3d.vertices)), torch.from_numpy(np.asarray(mesh_o3d.triangles))
            save_ply(os.path.join(mesh_dir, ply_path.split('/')[-1]), verts, faces)


def simplify_scans_2nd(dataset_dir, bin_path):
    dfaust_json = dfaust_utils.DFaustJson(bin_path)
    if not os.path.exists(os.path.join(dataset_dir, 'scans_simple_2nd')):
        os.mkdir(os.path.join(dataset_dir, 'scans_simple_2nd'))
    if not os.path.exists(os.path.join(dataset_dir, 'scans_simple_2nd', dfaust_json.data['subject'])):
        os.mkdir(os.path.join(dataset_dir, 'scans_simple_2nd', dfaust_json.data['subject']))

    for seq in tqdm(dfaust_json.data['seqs']):
        mesh_dir = os.path.join(dataset_dir, 'scans_simple_2nd', dfaust_json.data['subject'], seq['seq_name'])
        if not os.path.exists(mesh_dir):
            os.mkdir(mesh_dir)
        for i in tqdm(range(len(seq['frames']))):
            frame = seq['frames'][i]
            ply_path = os.path.join(dataset_dir, 'scans_simple', dfaust_json.data['subject'], seq['seq_name'], frame['ply_path'].split('/')[-1])
            verts, faces = load_ply(ply_path)
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(int(faces.shape[0] * 0.5))
            verts, faces = torch.from_numpy(np.asarray(mesh_o3d.vertices)), torch.from_numpy(np.asarray(mesh_o3d.triangles))
            save_ply(os.path.join(mesh_dir, ply_path.split('/')[-1]), verts, faces)


def filter_outlier_verts(ws_dir, dataset_dir, bin_path, config_path):
    dfaust_json = dfaust_utils.DFaustJson(bin_path)
    smpl_model = cape_utils.load_smpl(load_configs(config_path, ws_dir)).cuda()
    if not os.path.exists(os.path.join(dataset_dir, 'scans_simple')):
        os.mkdir(os.path.join(dataset_dir, 'scans_simple'))
    if not os.path.exists(os.path.join(dataset_dir, 'scans_simple', dfaust_json.data['subject'])):
        os.mkdir(os.path.join(dataset_dir, 'scans_simple', dfaust_json.data['subject']))

    for seq in tqdm(dfaust_json.data['seqs']):
        mesh_dir = os.path.join(dataset_dir, 'scans_simple', dfaust_json.data['subject'], seq['seq_name'])
        if not os.path.exists(mesh_dir):
            os.mkdir(mesh_dir)
        for i in tqdm(range(len(seq['frames']))):
            frame = seq['frames'][i]
            ply_path = os.path.join(dataset_dir, 'scans_simple', dfaust_json.data['subject'], seq['seq_name'], frame['ply_path'].split('/')[-1])
            verts, faces = load_ply(ply_path)
            verts, faces = verts.cuda(), faces.cuda()
            poses = torch.from_numpy(frame['poses']).float().cuda()[None]
            verts_smpl = smpl_model(poses).vertices[0]
            dst, _, _ = knn_points(verts[None], verts_smpl[None], K=1, return_nn=True)
            verts_mask = (dst.sqrt() < 0.1)[0, ..., 0]
            if (~verts_mask).sum().item() > 0:
                verts_mask = verts_mask.long()
                faces_mask = verts_mask[faces[:, 0]] * verts_mask[faces[:, 1]] * verts_mask[faces[:, 2]]
                faces_val = faces[faces_mask.bool()]
                verts_idxs_new2old = torch.arange(0, verts.shape[0]).long()[verts_mask.bool()]
                verts_idxs_old2new = torch.zeros_like(verts_mask) - 1
                verts_idxs_old2new[verts_idxs_new2old] = torch.arange(0, verts_idxs_new2old.shape[0]).long().cuda()
                faces = verts_idxs_old2new[faces_val]
                verts = verts[verts_idxs_new2old]
            save_ply(os.path.join(mesh_dir, ply_path.split('/')[-1]), verts, faces)


def save_registered_mesh(dataset_dir, subject, h5py_path):
    if not os.path.exists(os.path.join(dataset_dir, 'reg_meshes')):
        os.mkdir(os.path.join(dataset_dir, 'reg_meshes'))
    if not os.path.exists(os.path.join(dataset_dir, 'reg_meshes', subject)):
        os.mkdir(os.path.join(dataset_dir, 'reg_meshes', subject))

    seq_names = sorted(os.listdir(os.path.join(dataset_dir, 'scans', subject)))
    for seq_name in tqdm(seq_names):
        ply_names = sorted(os.listdir(os.path.join(dataset_dir, 'scans', subject, seq_name)))
        mesh_dir = os.path.join(dataset_dir, 'reg_meshes', subject, seq_name)
        if not os.path.exists(mesh_dir):
            os.mkdir(mesh_dir)
        sidseq = subject + '_' + seq_name
        with h5py.File(h5py_path, 'r') as f:
            if sidseq not in f:
                print('Sequence %s from subject %s not in %s' % (seq_name, subject, h5py_path))
                f.close()
                sys.exit(1)
            verts_seq = np.array(f[sidseq]).astype(np.float32).transpose([2, 0, 1])
            faces = np.array(f['faces']).astype(np.float32)
        for i, ply_name in tqdm(enumerate(ply_names)):
            verts = verts_seq[i]
            save_ply(os.path.join(mesh_dir, ply_name), torch.from_numpy(verts), torch.from_numpy(faces))


def add_idx(bin_path):
    dfaust_json = dfaust_utils.DFaustJson(bin_path)
    count = 0

    for seq in tqdm(dfaust_json.data['seqs']):
        for i in tqdm(range(len(seq['frames']))):
            frame = seq['frames'][i]
            frame['z_id'] = count
            count += 1

    dfaust_json.dump_bin_file(bin_path)
    print(count)


if __name__ == '__main__':
    # """
    # generate_DFaust_SMPLH('/mnt/ImpDyn_ws/DFaust',
    #                       '/mnt/ImpDyn_ws/SMPL',
    #                       '/mnt/ImpDyn_ws/DFaust/smplh_meshes',
    #                       '50002', 'male', gpu_id=0)
    # """
    # """
    # smplh_to_smpl('/mnt/ImpDyn_ws/DFaust',
    #               '50002',
    #               '/mnt/ImpDyn_ws/SMPL/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    # """
    # """
    # DFaust_parse_raw('/mnt/ImpDyn_ws/DFaust', '50002')
    # """
    # """
    # # 50002: interp (1st half train): seq_000_chicken_wings, seq_014_running_on_spot; extrap: seq_004_jumping_jacks, seq_015_shake_arms
    # split_train_test('/mnt/ImpDyn_ws/DFaust', 'v1', '/mnt/ImpDyn_ws/DFaust/50002_raw.bin', '50002',
    #                  ([0, 14], ['chicken_wings', 'running_on_spot']), ([4, 15], ['jumping_jacks', 'shake_arms']))
    # """
    # """
    # add_transl('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin', '50002',
    #            '/mnt/ImpDyn_ws/SMPL/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    # """
    # """
    # simplify_scans('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin', 'configs/DispInput/DFaust_50002/AutoRegr.yaml')
    # simplify_scans_2nd('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin')
    # """
    # """
    # filter_outlier_verts('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin', 'configs/DispInput/DFaust_50002/AutoRegr.yaml')
    # """
    # """
    # save_registered_mesh('/mnt/ImpDyn_ws/DFaust', '50002', '/mnt/ImpDyn_ws/DFaust/registrations_m.hdf5')
    # """
    # """
    # add_idx('/mnt/ImpDyn_ws/DFaust/50002_v1.bin')
    # """

    # # New process ---------------------
    # DFaust_smplh_to_smpl('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/SMPL', '/mnt/ImpDyn_ws/DFaust/smpl_poses', '50002', 'male', gpu_id=0)

    # DFaust_parse_raw('/mnt/ImpDyn_ws/DFaust', '50002')

    # # 50002: interp (1st half train): seq_000_chicken_wings, seq_014_running_on_spot; extrap: seq_009_one_leg_jump, seq_010_one_leg_jump
    # split_train_test('/mnt/ImpDyn_ws/DFaust', 'v2', '/mnt/ImpDyn_ws/DFaust/50002_raw.bin', '50002',
    #                  ([0, 14], ['chicken_wings', 'running_on_spot']), ([9, 10], ['one_leg_jump', 'one_leg_jump']))

    # """
    # simplify_scans('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin', 'configs/DispInput/DFaust_50002/AutoRegr.yaml')
    # simplify_scans_2nd('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin')
    # filter_outlier_verts('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin', 'configs/DispInput/DFaust_50002/AutoRegr.yaml')
    # save_registered_mesh('/mnt/ImpDyn_ws/DFaust', '50002', '/mnt/ImpDyn_ws/DFaust/registrations_m.hdf5')
    # """

    # add_idx('/mnt/ImpDyn_ws/DFaust/50002_v2.bin')

    # # ---------------------------------
    # """
    # generate_DFaust_SMPLH('/mnt/ImpDyn_ws/DFaust',
    #                       '/mnt/ImpDyn_ws/SMPL',
    #                       '/mnt/ImpDyn_ws/DFaust/smplh_meshes',
    #                       '50004', 'female', gpu_id=0)
    # """
    # """
    # smplh_to_smpl('/mnt/ImpDyn_ws/DFaust',
    #               '50004',
    #               '/mnt/ImpDyn_ws/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    # """
    # """
    # DFaust_parse_raw('/mnt/ImpDyn_ws/DFaust', '50004')
    # """
    # """
    # # 50004: interp (2nd half train): seq_000_chicken_wings, seq_014_running_on_spot; extrap: seq_004_jumping_jacks, seq_015_shake_arms
    # split_train_test('/mnt/ImpDyn_ws/DFaust', 'v1', '/mnt/ImpDyn_ws/DFaust/50004_raw.bin', '50004',
    #                  ([0, 18], ['chicken_wings', 'running_on_spot']), ([3, 19], ['jumping_jacks', 'shake_arms']))
    # """
    # """
    # add_transl('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50004_v1.bin', '50004',
    #            '/mnt/ImpDyn_ws/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    # """
    # """
    # simplify_scans('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50004_v1.bin', 'configs/DispInput/DFaust_50004/AutoRegr.yaml')
    # # simplify_scans_2nd('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin')
    # """

    # # New process ---------------------
    # DFaust_smplh_to_smpl('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/SMPL', '/mnt/ImpDyn_ws/DFaust/smpl_poses', '50004', 'female', gpu_id=0)

    # DFaust_parse_raw('/mnt/ImpDyn_ws/DFaust', '50004')

    # # 50004: interp (1st half train): seq_000_chicken_wings, seq_018_running_on_spot; extrap: seq_016_one_leg_loose
    # split_train_test('/mnt/ImpDyn_ws/DFaust', 'v2', '/mnt/ImpDyn_ws/DFaust/50004_raw.bin', '50004',
    #                  ([0, 18], ['chicken_wings', 'running_on_spot']), ([16], ['one_leg_loose']))

    # """
    # simplify_scans('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin', 'configs/DispInput/DFaust_50002/AutoRegr.yaml')
    # simplify_scans_2nd('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50002_v1.bin')
    # """
    # filter_outlier_verts('/mnt/ImpDyn_ws/DFaust', '/mnt/ImpDyn_ws/DFaust/50004_v2.bin', 'configs/DispInput/DFaust_50004/AutoRegr.yaml')
    # """
    # save_registered_mesh('/mnt/ImpDyn_ws/DFaust', '50004', '/mnt/ImpDyn_ws/DFaust/registrations_f.hdf5')
    # """

    # add_idx('/mnt/ImpDyn_ws/DFaust/50004_v2.bin')


    from argparse import ArgumentParser

    parser = ArgumentParser(description='Process DFaust data.')
    parser.add_argument('--ws_dir', required=True, help='path of work space directory')
    args = parser.parse_args()

    # New process ---------------------
    DFaust_smplh_to_smpl(
        os.path.join(args.ws_dir, 'DFaust'),
        os.path.join(args.ws_dir, 'SMPL'),
        os.path.join(args.ws_dir, 'DFaust', 'smpl_poses'),
        '50002', 'male', gpu_id=0
    )

    DFaust_parse_raw(os.path.join(args.ws_dir, 'DFaust'), '50002')

    # 50002: interp (1st half train): seq_000_chicken_wings, seq_014_running_on_spot; extrap: seq_009_one_leg_jump, seq_010_one_leg_jump
    split_train_test(
        os.path.join(args.ws_dir, 'DFaust'),
        'v2',
        os.path.join(args.ws_dir, 'DFaust', '50002_raw.bin'),
        '50002',
        ([0, 14], ['chicken_wings', 'running_on_spot']),
        ([9, 10], ['one_leg_jump', 'one_leg_jump'])
    )

    simplify_scans(
        args.ws_dir,
        os.path.join(args.ws_dir, 'DFaust'),
        os.path.join(args.ws_dir, 'DFaust', '50002_v2.bin'),
        'configs/PosedDecKNN_dPoses_dHs/AutoRegr.yaml'
    )

    filter_outlier_verts(
        args.ws_dir,
        os.path.join(args.ws_dir, 'DFaust'),
        os.path.join(args.ws_dir, 'DFaust', '50002_v2.bin'),
        'configs/PosedDecKNN_dPoses_dHs/AutoRegr.yaml'
    )

    # save_registered_mesh('/mnt/ImpDyn_ws/DFaust', '50002', '/mnt/ImpDyn_ws/DFaust/registrations_m.hdf5')

    add_idx(os.path.join(args.ws_dir, 'DFaust', '50002_v2.bin'))

    # ---------------------------------
