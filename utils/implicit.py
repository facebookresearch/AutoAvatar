# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# import openvdb as vdb
import numpy as np
import os
import torch
import torch.nn.functional as F
import math
from skimage import measure


def build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1):
    smooth_conv = torch.nn.Conv3d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, padding=padding
    )
    smooth_conv.weight.data = torch.ones(
        (kernel_size, kernel_size, kernel_size), 
        dtype=torch.float32
    ).reshape(in_channels, out_channels, kernel_size, kernel_size, kernel_size) / (kernel_size**3)
    smooth_conv.bias.data = torch.zeros(out_channels)
    return smooth_conv

def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None, thresh=0.5, texture_net = None, poses=None, shapes=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # Then we define the lambda function for cell evaluation
    color_flag = False if texture_net is None else True

    def eval_func(points):
        samples = points.t().unsqueeze(0).to(cuda)
        # pred = net.query(samples, calib_tensor)[0][0]
        pred = net(samples, poses, shapes)[0]
        return pred

    def batch_eval(points, num_samples=num_samples):
        num_pts = points.shape[1]
        sdf = []
        num_batches = num_pts // num_samples
        for i in range(num_batches):
            sdf.append(
                eval_func(points[:, i * num_samples:i * num_samples + num_samples])
            )
        if num_pts % num_samples:
            sdf.append(
                eval_func(points[:, num_batches * num_samples:])
            )
        if num_pts == 0:
            return None
        sdf = torch.cat(sdf)
        return sdf

    # Then we evaluate the grid    
    max_level = int(math.log2(resolution))
    sdf = eval_progressive(batch_eval, 4, max_level, cuda, b_min, b_max, thresh)

    # calculate matrix
    mat = np.eye(4)
    length = b_max - b_min
    mat[0, 0] = length[0] / sdf.shape[0]
    mat[1, 1] = length[1] / sdf.shape[1]
    mat[2, 2] = length[2] / sdf.shape[2]
    mat[0:3, 3] = b_min

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, thresh, gradient_direction='ascent')
    except:
        print('error cannot marching cubes')
        return -1
    # grid = vdb.FloatGrid(1.0)
    # grid.copyFromArray(sdf)
    # verts, quads = grid.convertToQuads()
    # faces = np.zeros((quads.shape[0] * 2, 3), dtype=np.uint32)
    # faces[:quads.shape[0], :] = quads[:, [0, 2, 1]]
    # faces[quads.shape[0]:, :] = quads[:, [0, 3, 2]]
    # verts = np.zeros((10, 3), dtype=np.float32)
    # faces = np.zeros((10, 3), dtype=np.int32)

    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    if np.linalg.det(mat) > 0:
        faces = faces[:,[0,2,1]]

    if color_flag:
        torch_verts = torch.Tensor(verts).unsqueeze(0).permute(0,2,1).to(cuda)

        with torch.no_grad():
            _, last_layer_feature, point_local_feat = net.query(torch_verts, calib_tensor, return_last_layer_feature=True)
            vertex_colors = texture_net.query(point_local_feat, last_layer_feature)
            vertex_colors = vertex_colors.squeeze(0).permute(1,0).detach().cpu().numpy()
        return verts, faces, vertex_colors #, normals, values, vertex_colors
    else:
        return verts, faces #, normals, values


def eval_progressive(batch_eval, min_level, max_level, cuda, b_min, b_max, thresh=0.5):
    steps = [i for i in range(min_level, max_level+1)]

    b_min = torch.tensor(b_min).to(cuda)
    b_max = torch.tensor(b_max).to(cuda)

    # init
    smooth_conv3x3 = build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1).to(cuda)

    arrange = torch.linspace(0, 2**steps[-1], 2**steps[0]+1).long().to(cuda)
    coords = torch.stack(torch.meshgrid([
        arrange, arrange, arrange
    ])) # [3, 2**step+1, 2**step+1, 2**step+1]
    coords = coords.view(3, -1).t() # [N, 3]
    calculated = torch.zeros(
        (2**steps[-1]+1, 2**steps[-1]+1, 2**steps[-1]+1), dtype=torch.bool
    ).to(cuda)
        
    gird8_offsets = torch.stack(torch.meshgrid([
        torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])
    ])).int().to(cuda).view(3, -1).t() #[27, 3]

    with torch.no_grad():
        for step in steps:
            resolution = 2**step + 1
            stride = 2**(steps[-1]-step)

            if step == steps[0]:
                coords2D = coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min
                sdf_all = batch_eval(
                    coords2D.t(),
                ).view(resolution, resolution, resolution)
                coords_accum = coords / stride
                coords_accum = coords_accum.long()
                calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

            else:
                valid = F.interpolate(
                    (sdf_all>thresh).view(1, 1, *sdf_all.size()).float(), 
                    size=resolution, mode="trilinear", align_corners=True
                )[0, 0]
                
                sdf_all = F.interpolate(
                    sdf_all.view(1, 1, *sdf_all.size()), 
                    size=resolution, mode="trilinear", align_corners=True
                )[0, 0]

                coords_accum *= 2

                is_boundary = (valid > 0.0) & (valid < 1.0)
                is_boundary = smooth_conv3x3(is_boundary.float().view(1, 1, *is_boundary.size()))[0, 0] > 0

                is_boundary[coords_accum[:, 0], coords_accum[:, 1], coords_accum[:, 2]] = False

                # coords = is_boundary.nonzero() * stride
                coords = torch.nonzero(is_boundary) * stride
                coords2D = coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min
                # coords2D = coords.float() / (2**steps[-1]+1)
                sdf = batch_eval(
                    coords2D.t(), 
                ) #[N]
                if sdf is None:
                    continue
                if sdf is not None:
                    sdf_all[is_boundary] = sdf
                voxels = coords / stride
                voxels = voxels.long()
                coords_accum = torch.cat([
                    voxels, 
                    coords_accum
                ], dim=0).unique(dim=0)
                calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

                for n_iter in range(14):
                    sdf_valid = valid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
                    idxs_danger = ((sdf_valid==1) & (sdf<thresh)) | ((sdf_valid==0) & (sdf>thresh)) #[N,]
                    coords_danger = coords[idxs_danger, :] #[N, 3]
                    if coords_danger.size(0) == 0:
                        break

                    coords_arround = coords_danger.int() + gird8_offsets.view(-1, 1, 3) * stride
                    coords_arround = coords_arround.reshape(-1, 3).long()
                    coords_arround = coords_arround.unique(dim=0)
                    
                    coords_arround[:, 0] = coords_arround[:, 0].clamp(0, calculated.size(0)-1)
                    coords_arround[:, 1] = coords_arround[:, 1].clamp(0, calculated.size(1)-1)
                    coords_arround[:, 2] = coords_arround[:, 2].clamp(0, calculated.size(2)-1)

                    coords = coords_arround[
                        calculated[coords_arround[:, 0], coords_arround[:, 1], coords_arround[:, 2]] == False
                    ]
                    
                    if coords.size(0) == 0:
                        break
                    
                    coords2D = coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min
                    # coords2D = coords.float() / (2**steps[-1]+1)
                    sdf = batch_eval(
                        coords2D.t(), 
                    ) #[N]

                    voxels = coords / stride
                    voxels = voxels.long()
                    sdf_all[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = sdf
                    
                    coords_accum = torch.cat([
                        voxels, 
                        coords_accum
                    ], dim=0).unique(dim=0)
                    calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

        return sdf_all.data.cpu().numpy()
