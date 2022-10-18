# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import (
    PerspectiveCameras,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex,
    rasterize_meshes
)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import rasterize_meshes
from numba import jit
import copy
import open3d as o3d


# Functions -----------------------------------------------------------------------------------------------------
def render_mesh(verts, faces, R, t, f, image_size=(512, 512), colors=None, simplify_mesh=False):
    """
    :param verts: (N, 3)
    :param faces: (F, 3)
    """
    device = verts.device
    f_th = torch.tensor(f, dtype=torch.float32, device=device)[None]
    image_size_th = torch.tensor(image_size, dtype=torch.int32, device=device)[None]
    cameras = PerspectiveCameras(focal_length=f_th, R=R[None], T=t[None], device=device, image_size=image_size_th)
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    lights = AmbientLights(device=device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    if not simplify_mesh:
        if colors is not None:
            mesh = Meshes(verts=verts[None], faces=faces[None], textures=TexturesVertex(colors[None]))
        else:
            mesh = Meshes(verts=verts[None], faces=faces[None])
            normals = (mesh.verts_normals_padded() + 1) / 2
            mesh = Meshes(verts=verts[None], faces=faces[None], textures=TexturesVertex(normals))
    else:
        if colors is None:
            mesh = Meshes(verts=verts[None], faces=faces[None])
            normals = (mesh.verts_normals_padded() + 1) / 2
            colors = normals[0]
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(int(faces.shape[0] * 0.1))
        verts, faces, colors = torch.from_numpy(np.asarray(mesh_o3d.vertices)), torch.from_numpy(np.asarray(mesh_o3d.triangles)), torch.from_numpy(np.asarray(mesh_o3d.vertex_colors))
        mesh = Meshes(verts=verts[None].float(), faces=faces[None], textures=TexturesVertex(colors[None].float())).to(device)

    images = renderer(mesh)[0, ..., :3].clip(min=0, max=1)
    return images


def parse_uv_info(obj_path):
    verts, faces_tuple, aux_tuple = load_obj(obj_path)
    faces = faces_tuple.verts_idx.numpy()
    faces_uv = faces_tuple.textures_idx.numpy()
    verts_uv = aux_tuple.verts_uvs.numpy()
    verts_uv = verts_uv * 2 - 1 #(1 - verts_uv) * 2 - 1
    N = verts.shape[0]
    F = faces.shape[0]
    M = verts_uv.shape[0]
    assert faces_uv.shape == (F, 3)
    print(N, F, M)

    v2uv = np.zeros((N, 10), dtype=np.int32) - 1
    v2uv_count = np.zeros((N,), dtype=np.int32)
    @jit(nopython=True)
    def func(faces, faces_uv, v2uv, v2uv_count):
        for i in range(F):
            for k in range(3):
                v = faces[i, k]
                uv = faces_uv[i, k]
                included = False
                for j in range(10):
                    if v2uv[v, j] == uv:
                        included = True
                        break
                if not included:
                    v2uv[v, v2uv_count[v]] = uv
                    v2uv_count[v] += 1
        for i in range(N):
            for k in range(10):
                if v2uv[i, k] == -1:
                    v2uv[i, k] = v2uv[i, 0]
        return v2uv, v2uv_count

    v2uv, v2uv_count = func(faces, faces_uv, v2uv, v2uv_count)
    print(np.amin(v2uv_count), np.amax(v2uv_count))
    v2uv = v2uv[:, :np.amax(v2uv_count)]

    return verts_uv, faces_uv, v2uv, faces


def compute_per_pixel_verts_idx_bary_weights(verts_uv, faces_uv, v2uv, uv_size):
    # Compute uv2v
    N, K = v2uv.shape
    M = verts_uv.shape[0]
    uv2v = torch.zeros((M,), dtype=torch.long) - 1
    for i in range(K):
        uv2v[v2uv[:, i]] = torch.arange(N)

    # Rasterization
    verts_uv = -verts_uv
    verts_uv_ = torch.cat([verts_uv, torch.ones((M, 1), dtype=torch.float)], dim=-1)
    meshes = Meshes(verts=verts_uv_[None].cuda(), faces=faces_uv[None].cuda())
    pix_to_face, _, barycentric, _ = rasterize_meshes(meshes, uv_size, faces_per_pixel=1) #, blur_radius=0.0001, clip_barycentric_coords=True)
    assert pix_to_face.shape == (1, uv_size, uv_size, 1) and barycentric.shape == (1, uv_size, uv_size, 1, 3)
    faces_uv_ = torch.cat([-torch.ones((1, 3), dtype=torch.long), faces_uv], dim=0)     # (1 + F, 3)
    pix_to_uv = faces_uv_[pix_to_face[0, ..., 0] + 1]
    assert pix_to_uv.shape == (uv_size, uv_size, 3)
    uv2v_ = torch.cat([-torch.ones((1,), dtype=torch.long), uv2v], dim=0)               # (1 + M,)
    pix_to_v = uv2v_[pix_to_uv + 1]
    assert pix_to_v.shape == (uv_size, uv_size, 3)

    return pix_to_v, barycentric[0, ..., 0, :]


# Classes -------------------------------------------------------------------------------------------------------
class UVRender(nn.Module):
    def __init__(self, args, verts_uv, faces_uv, v2uv):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.register_buffer('verts_uv', verts_uv)
        self.register_buffer('faces_uv', faces_uv)
        self.register_buffer('v2uv', v2uv)

        pix_to_v, bary_w = compute_per_pixel_verts_idx_bary_weights(verts_uv, faces_uv, v2uv, args['model']['uv_size'])
        self.register_buffer('pix_to_v', pix_to_v)
        self.register_buffer('bary_w', bary_w)

    def to_uv(self, verts):
        """
        :param verts: (B, N, C)
        """
        B, N, C = verts.shape
        verts_ = torch.cat([torch.zeros((B, 1, C), dtype=torch.float, device=verts.device), verts], dim=1)  # (B, 1 + N, C)
        pix_verts = verts_[:, self.pix_to_v + 1, :]                                         # (B, H, W, 3, C)
        verts_uv = (pix_verts * self.bary_w[None, ..., None]).sum(dim=-2)                   # (B, H, W, C)
        assert verts_uv.shape == (B, self.args['model']['uv_size'], self.args['model']['uv_size'], C)
        return verts_uv.permute(0, 3, 1, 2).contiguous()

    def from_uv(self, verts_uv):
        """
        :param verts_uv: (B, C, H, W)
        """
        B, C, H, W = verts_uv.shape
        N, K = self.v2uv.shape
        grid = self.verts_uv[self.v2uv][None].expand(B, N, K, 2).contiguous()
        verts = F.grid_sample(verts_uv, grid, mode='bilinear', align_corners=False)         # (B, C, N, K)
        assert verts.shape == (B, C, N, K)
        verts = verts.mean(dim=-1).permute(0, 2, 1).contiguous()
        return verts
