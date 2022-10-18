# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F
from pytorch3d.ops import norm_laplacian, sample_points_from_meshes, knn_points, knn_gather
from pytorch3d.structures import Meshes, Pointclouds, utils as struct_utils
from pytorch3d.io import load_ply
from pytorch3d import _C
from pytorch3d.renderer import TexturesVertex
from pytorch3d.transforms import axis_angle_to_quaternion
import smplx
from smplx.utils import SMPLOutput
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform


# Functions -----------------------------------------------------------------------------------------------------
def load_smpl(args):
    if args['data']['type'] == 'CAPE':
        ply_path = os.path.join(args['data']['raw_dataset_dir'], 'minimal_body_shape', args['data']['subject'], '%s_minimal.ply' % args['data']['subject'])
    elif args['data']['type'] == 'DFaust':
        ply_path = os.path.join(args['data']['dataset_dir'], 'smpl_poses', args['data']['subject'], 'v_template.ply')
    v_template, _ = load_ply(ply_path)
    smpl_model = MySMPL(args['data']['smpl_path'], v_template=v_template)
    return smpl_model


def taubin_smoothing(
    meshes: Meshes, lambd: float = 0.53, mu: float = -0.53, num_iter: int = 10
) -> Meshes:
    """
    Taubin smoothing [1] is an iterative smoothing operator for meshes.
    At each iteration
        verts := (1 - λ) * verts + λ * L * verts
        verts := (1 - μ) * verts + μ * L * verts

    This function returns a new mesh with smoothed vertices.
    Args:
        meshes: Meshes input to be smoothed
        lambd, mu: float parameters for Taubin smoothing,
            lambd > 0, mu < 0
        num_iter: number of iterations to execute smoothing
    Returns:
        mesh: Smoothed input Meshes

    [1] Curve and Surface Smoothing without Shrinkage,
        Gabriel Taubin, ICCV 1997
    """
    verts = meshes.verts_packed()  # V x 3
    edges = meshes.edges_packed()  # E x 3

    for _ in range(num_iter):
        L = norm_laplacian(verts, edges)
        total_weight = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        verts = (1 - lambd) * verts + lambd * torch.mm(L, verts) / (total_weight + 1e-10)

        # pyre-ignore
        L = norm_laplacian(verts, edges)
        total_weight = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        verts = (1 - mu) * verts + mu * torch.mm(L, verts) / (total_weight + 1e-10)

    verts_list = struct_utils.packed_to_list(
        verts, meshes.num_verts_per_mesh().tolist()
    )
    mesh = Meshes(verts=list(verts_list), faces=meshes.faces_list())
    return mesh


def compute_adjacent_matrix(parents, n_rings):
    """
    :param parents: (J,)
    """
    J = parents.shape[0]
    W = torch.zeros(J, J - 1)
    for i in range(J - 1):
        W[i + 1, i] += 1.0
        parent = parents[i+1]
        for j in range(n_rings):
            W[parent, i] += 1.0
            if parent == 0:
                break
            parent = parents[parent]
    # W /= W.sum(0, keepdim=True) + 1e-16
    return W


def sample_igr_pts(verts, faces, bbmin, bbmax, args):
    """
    :param verts: (B, N, 3)
    :param faces: (B, F, 3)
    :param bbmin / bbmax: (B, 3)
    """
    B, N, _ = verts.shape
    meshes = Meshes(verts=verts, faces=faces)
    if not args['model']['use_detail']:
        surf_pts, surf_normals = sample_points_from_meshes(meshes, num_samples=args['train']['n_pts_scan'], return_normals=True)
    else:
        normals = meshes.verts_normals_padded()
        meshes = Meshes(verts=verts, faces=faces, textures=TexturesVertex(normals))
        surf_pts, surf_normals = sample_points_from_meshes(meshes, num_samples=args['train']['n_pts_scan'], return_textures=True)
        surf_normals = F.normalize(surf_normals, p=2, dim=-1)
    igr_pts = surf_pts[:, :args['train']['n_pts_scan_igr']] + torch.normal(0, args['train']['pts_igr_sigma'], (B, args['train']['n_pts_scan_igr'], 3), device=verts.device)
    igr_pts = torch.minimum(torch.maximum(igr_pts, bbmin[:, None].expand(B, args['train']['n_pts_scan_igr'], 3)), bbmax[:, None].expand(B, args['train']['n_pts_scan_igr'], 3))
    bbox_pts = torch.rand((B, args['train']['n_pts_bbox_igr'], 3), device=verts.device) * (bbmax - bbmin)[:, None] + bbmin[:, None]
    rand_pts = torch.cat([igr_pts, bbox_pts], dim=1)
    return surf_pts, surf_normals, igr_pts, bbox_pts, rand_pts


def _point_to_edge_distance(
    point: torch.Tensor, s0, s1
) -> torch.Tensor:
    """
    Computes the squared euclidean distance of points to edges. Modified from https://github.com/facebookresearch/pytorch3d/issues/613
    Args:
        point: FloatTensor of shape (P, 3)
        edge: FloatTensor of shape (P, 2, 3)
    Returns:
        dist: FloatTensor of shape (P,)
        x: FloatTensor of shape (P, 3)
    If a, b are the start and end points of the segments, we
    parametrize a point p as
        x(t) = a + t * (b - a)
    To find t which describes p we minimize (x(t) - p) ^ 2
    Note that p does not need to live in the space spanned by (a, b)
    """
    s01 = s1 - s0
    norm_s01 = (s01 * s01).sum(dim=-1)

    same_edge = norm_s01 < 1e-8
    t = torch.where(same_edge, torch.ones_like(norm_s01) * 0.5, (s01 * (point - s0)).sum(dim=-1) / norm_s01)
    t = torch.clamp(t, min=0.0, max=1.0)[..., None]
    x = s0 + t * s01
    dist = ((x - point) * (x - point)).sum(dim=-1).sqrt()
    return dist, x


def _point_to_bary(point: torch.Tensor, a, b, c) -> torch.Tensor:
    """
    Computes the barycentric coordinates of point wrt triangle (tri)
    Note that point needs to live in the space spanned by tri = (a, b, c),
    i.e. by taking the projection of an arbitrary point on the space spanned by
    tri. Modified from https://github.com/facebookresearch/pytorch3d/issues/613
    Args:
        point: FloatTensor of shape (P, 3)
        tri: FloatTensor of shape (3, 3)
    Returns:
        bary: FloatTensor of shape (P, 3)
    """
    assert point.dim() == 2 and point.shape[1] == 3
    P, _ = point.shape
    assert a.shape == (P, 3) and b.shape == (P, 3) and c.shape == (P, 3)

    v0 = b - a
    v1 = c - a
    v2 = point - a

    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0).sum(dim=-1)
    d21 = (v2 * v1).sum(dim=-1)

    denom = d00 * d11 - d01 * d01 + 1e-8
    s2 = (d11 * d20 - d01 * d21) / denom
    s3 = (d00 * d21 - d01 * d20) / denom
    s1 = 1.0 - s2 - s3

    bary = torch.stack([s1, s2, s3], dim=-1)
    return bary


def proj_pts_to_mesh(pts, verts, faces, verts_feat=None, scale=1000, return_idxs=False):
    """
    :param pts: (B, M, 3)
    :param verts: (B, N, 3)
    :param faces: (B, F, 3)
    :param verts_feat: (B, N, C)
    """
    B, M, _ = pts.shape
    N = verts.shape[1]
    F = faces.shape[1]
    pts = pts * scale
    verts = verts * scale
    meshes = Meshes(verts=verts, faces=faces)
    pcls = Pointclouds(pts)
    assert len(meshes) == B and len(pcls) == B

    # packed representation for pointclouds
    points = pcls.points_packed()           # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()
    assert torch.allclose(points, pts.view(-1, 3))

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]       # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()
    verts_normals_packed = meshes.verts_normals_packed()
    faces_normals_packed = meshes.faces_normals_packed()
    assert torch.allclose(verts_packed, verts.view(-1, 3)) #and torch.allclose(faces_packed, faces.view(-1, 3))

    dists, idxs = _C.point_face_dist_forward(
        points, points_first_idx, tris, tris_first_idx, max_points, 1e-3
    )
    pts_faces_normals = faces_normals_packed[idxs]                  # (P, 3)
    pts_verts_normals = verts_normals_packed[faces_packed][idxs]    # (P, 3, 3)
    pts_tris = tris[idxs]                                           # (P, 3, 3)

    # Project pts to the plane of its closest triangle
    v, v0, v1, v2 = points, pts_tris[:, 0], pts_tris[:, 1], pts_tris[:, 2]
    sd = -((v0 - v) * pts_faces_normals).sum(dim=-1, keepdim=True)
    v_proj = -sd * pts_faces_normals + v

    # Check v_proj outside triangle
    inside = torch.isclose(sd[:, 0].abs(), dists.sqrt(), atol=1e-5)
    outside = torch.logical_not(inside)

    # Project pts to triangle edges
    if outside.sum().item() > 0:
        e01_dist, e01_v_proj = _point_to_edge_distance(v[outside], v0[outside], v1[outside])
        e02_dist, e02_v_proj = _point_to_edge_distance(v[outside], v0[outside], v2[outside])
        e12_dist, e12_v_proj = _point_to_edge_distance(v[outside], v1[outside], v2[outside])
        e_dist = torch.stack([e01_dist, e02_dist, e12_dist], dim=0)             # (3, P_)
        e_v_proj = torch.stack([e01_v_proj, e02_v_proj, e12_v_proj], dim=0)     # (3, P_, 3)
        e_min_idxs = torch.argmin(e_dist, dim=0)                                # (P_,)
        v_proj_out = torch.gather(e_v_proj, dim=0, index=e_min_idxs[None, :, None].expand(1, e_dist.shape[1], 3))[0]
        v_proj[outside] = v_proj_out

    # Compute barycentric coordinates
    bary = _point_to_bary(v_proj, v0, v1, v2)        # (P, 3)
    pts_normals = (pts_verts_normals * bary[..., None]).sum(dim=-2)
    sd = torch.norm(v - v_proj + 1e-8, dim=-1, p=2) * ((v - v_proj) * pts_normals).sum(dim=-1).sign()

    # Test
    if not torch.allclose(sd.abs(), dists.sqrt(), atol=1e-3):
        print('sd:', (sd.abs() - dists.sqrt()).abs().max(), ((sd.abs() - dists.sqrt()).abs() > 1e-3).sum())
    # v_proj_rec = (pts_tris * bary[..., None]).sum(dim=-2)
    # if not torch.allclose(v_proj, v_proj_rec, atol=1e-3):
    #     print('v_proj:', (v_proj - v_proj_rec).abs().max())
    # if sd.isnan().sum().item() > 0:
        # print(sd.isnan().sum(), '/', sd.shape)

    if verts_feat is not None:
        C = verts_feat.shape[-1]
        assert verts_feat.shape == (B, N, C)
        verts_feat_packed = verts_feat.view(-1, C)
        pts_verts_feat = verts_feat_packed[faces_packed][idxs]      # (P, 3, C)
        pts_feat = (pts_verts_feat * bary[..., None]).sum(dim=-2)
        pts_feat = pts_feat.view(B, M, C)
    else:
        pts_feat = None

    if not return_idxs:
        return sd.view(B, M) / scale, v_proj.view(B, M, 3) / scale, faces_packed[idxs].reshape(B, M, 3), bary.view(B, M, 3), pts_feat
    else:
        return sd.view(B, M) / scale, v_proj.view(B, M, 3) / scale, faces_packed[idxs].reshape(B, M, 3), bary.view(B, M, 3), pts_feat, idxs.view(B, M)


def proj_pts_to_mesh_sample(pts, verts, faces, verts_feat=None, n_sample=100000):
    """
    :param pts: (B, M, 3)
    :param verts: (B, N, 3)
    :param faces: (B, F, 3)
    :param verts_feat: (B, N, C)
    """
    B, M, _ = pts.shape
    F = faces.shape[1]
    K = n_sample
    if verts_feat is None:
        verts_feat = torch.zeros_like(verts)
    C = verts_feat.shape[-1]
    meshes = Meshes(verts=verts, faces=faces, textures=TexturesVertex(verts_feat))
    pts_v, pts_v_normals, pts_v_feat = sample_points_from_meshes(meshes, num_samples=n_sample, return_normals=True, return_textures=True)
    assert pts_v.shape == (B, K, 3) and pts_v_normals.shape == (B, K, 3) and pts_v_feat.shape == (B, K, C)

    # KNN
    _, idx, nn = knn_points(pts, pts_v, K=1, return_nn=True)
    assert torch.allclose(nn, knn_gather(pts_v, idx)) and idx.shape == (B, M, 1)
    nn_normals = knn_gather(pts_v_normals, idx)
    nn_feat = knn_gather(pts_v_feat, idx)
    assert nn.shape == (B, M, 1, 3) and nn_normals.shape == (B, M, 1, 3) and nn_feat.shape == (B, M, 1, C)
    nn, nn_normals, nn_feat = nn[:, :, 0], nn_normals[:, :, 0], nn_feat[:, :, 0]

    sd = torch.norm(pts - nn + 1e-8, dim=-1, p=2) * ((pts - nn) * nn_normals).sum(dim=-1).sign()
    return sd, nn, nn_normals, nn_feat


def compute_signed_dst(pts, verts, faces, scale=1000):
    """
    :param pts: (B, M, 3)
    :param verts: (B, N, 3)
    :param faces: (B, F, 3)
    """
    B, M, _ = pts.shape
    F = faces.shape[1]
    pts = pts * scale
    verts = verts * scale
    meshes = Meshes(verts=verts, faces=faces)
    pcls = Pointclouds(pts)
    assert len(meshes) == B and len(pcls) == B

    # packed representation for pointclouds
    points = pcls.points_packed()           # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()
    assert torch.allclose(points, pts.view(-1, 3))

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]       # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()
    verts_normals_packed = meshes.verts_normals_packed()
    faces_normals_packed = meshes.faces_normals_packed()
    assert torch.allclose(verts_packed, verts.view(-1, 3)) and torch.allclose(faces_packed, faces.view(-1, 3))

    dists, idxs = _C.point_face_dist_forward(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    pts_faces_normals = faces_normals_packed[idxs]                  # (P, 3)
    pts_verts_normals = verts_normals_packed[faces_packed][idxs]    # (P, 3, 3)
    pts_tris = tris[idxs]                                           # (P, 3, 3)

    verts_normals = meshes.verts_normals_padded()
    _, nn, _, nn_normals = proj_pts_to_mesh_sample(pts, verts, faces, verts_feat=verts_normals, n_sample=100000)
    sd = dists.sqrt() * ((pts - nn) * nn_normals).sum(dim=-1).sign()
    return sd.view(B, M) / scale


def scan_to_pred_errors(verts_scan, faces_scan, verts_pred, faces_pred):
    """
    :param verts_scan: (B, N_s, 3)
    :param faces_scan: (B, F_s, 3)
    :param verts_pred: (B, N_p, 3)
    :param faces_pred: (B, F_p, 3)
    """
    B, N_s, _ = verts_scan.shape
    N_p = verts_pred.shape[1]
    assert verts_scan.shape == (B, N_s, 3) and verts_pred.shape == (B, N_p, 3)
    meshes = Meshes(verts=verts_scan, faces=faces_scan)
    normals_scan = meshes.verts_normals_padded()
    meshes = Meshes(verts=verts_pred, faces=faces_pred)
    normals_pred = meshes.verts_normals_padded()
    assert normals_pred.shape == (B, N_p, 3)
    sd_err, _, _, _, normals_proj = proj_pts_to_mesh(verts_scan, verts_pred, faces_pred, normals_pred)
    cos_err = F.cosine_similarity(normals_scan, normals_proj, dim=-1)
    assert sd_err.shape == (B, N_s) and cos_err.shape == (B, N_s)
    return sd_err, cos_err


def proj_pts_to_uv(pts, verts, faces, verts_uv, faces_uv, uv_feat=None):
    """
    :param pts: (B, M, 3)
    :param verts: (B, N, 3)
    :param faces: (B, F, 3)
    :param verts_uv: (B, N_, 2)
    :param faces_uv: (B, F, 3)
    :param uv_feat: (B, C, H, W)
    """
    B, M, _ = pts.shape
    N = verts.shape[1]
    F_ = faces.shape[1]
    N_ = verts_uv.shape[1]
    assert pts.shape == (B, M, 3) and verts.shape == (B, N, 3) and faces.shape == (B, F_, 3) and verts_uv.shape == (B, N_, 2) and faces_uv.shape == (B, F_, 3)
    sd, v_proj, _, bary_w, _, pts_faces_idxs = proj_pts_to_mesh(pts, verts, faces, return_idxs=True)

    pts_faces_idxs_packed = pts_faces_idxs.view(B * M,)     # (P,)
    verts_uv_ = torch.cat([verts_uv, torch.zeros_like(verts_uv[:, :, :1])], dim=-1)     # (B, N_, 3)
    meshes = Meshes(verts=verts_uv_, faces=faces_uv)
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    pts_verts_uv = verts_packed[faces_packed][pts_faces_idxs_packed][:, :, :2]          # (P, 3, 2)
    pts_uv = (pts_verts_uv * bary_w.view(B * M, 3, 1)).sum(dim=-2)
    pts_uv = pts_uv.view(B, M, 1, 2)

    _, C, H, W = uv_feat.shape
    assert uv_feat.shape == (B, C, H, W)
    # pts_feat = F.grid_sample(uv_feat, pts_uv, mode='bilinear', align_corners=False)     # (B, C, M, 1)
    grid_sample = MyGridSample.apply
    pts_feat = grid_sample(pts_uv, uv_feat)         # (B, C, M, 1)
    assert pts_feat.shape == (B, C, M, 1)
    pts_feat = pts_feat.permute(0, 2, 1, 3).squeeze(-1).contiguous()
    assert pts_feat.shape == (B, M, C)

    return sd, v_proj, pts_feat, pts_uv.view(B, M, 2)


def lbs(
    betas,
    pose,
    v_template,
    shapedirs,
    posedirs,
    J_regressor,
    parents,
    lbs_weights,
    pose2rot: bool = True,
):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A


# Classes -------------------------------------------------------------------------------------------------------
class CAPEJson():
    """
    CAPE .bin Structure:
        'subject'
        'cloth_type'
        'seqs'
        for seq in seqs:
            'id'
            'seq_name': longlong_athletics_trial1
            'frames'
            for frame in frames:
                'npz_path'
                'smooth_mesh_path': New field!!!
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

    def append_frames(self, frames, npz_path):
        frames.append({
            'npz_path': npz_path
        })
        return frames

    def append_seqs(self, seqs, seq_name, frames):
        seqs.append({
            'id': len(seqs),
            'seq_name': seq_name,
            'frames': frames
        })
        return seqs

    def set_data(self, subject, cloth_type, seqs):
        self.data = {
            'subject': subject,
            'cloth_type': cloth_type,
            'seqs': seqs
        }

    def num_of_seqs(self):
        return len(self.data['seqs'])
    
    def num_of_frames(self):
        count = 0
        for seq in self.data['seqs']:
            count += len(seq['frames'])
        return count


class MySMPL(smplx.SMPLLayer):
    def __init__(
        self, model_path: str,
	    kid_template_path: str = '',
        data_struct = None,
        create_betas: bool = True,
        betas = None,
        num_betas: int = 10,
        create_global_orient: bool = True,
        global_orient = None,
        create_body_pose: bool = True,
        body_pose = None,
        create_transl: bool = True,
        transl = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        gender: str = 'neutral',
        age: str = 'adult',
        vertex_ids = None,
        v_template = None,
        **kwargs
    ) -> None:
        super().__init__(model_path=model_path, kid_template_path=kid_template_path, data_struct=data_struct, betas=betas, num_betas=num_betas,
                         global_orient=global_orient, body_pose=body_pose, transl=transl, dtype=dtype, batch_size=batch_size, joint_mapper=joint_mapper, 
                         gender=gender, age=age, vertex_ids=vertex_ids, v_template=v_template, **kwargs)
        self.register_buffer('pose_cano', torch.zeros((1, self.NUM_BODY_JOINTS * 3), dtype=dtype))
        self.faces = self.faces_tensor

    def forward(
        self,
        poses,
        betas = None,
        body_pose = None,
        global_orient = None,
        transl = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SMPLOutput:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        transl, global_orient, body_pose = poses[:, :3], poses[:, 3:6], poses[:, 6:]
        apply_trans = True
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        batch_size = poses.shape[0]
        betas = torch.zeros([batch_size, self.num_betas], dtype=self.dtype, device=poses.device)

        vertices, joints, A = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None)
        output.A = A

        return output

    @classmethod
    def compute_poses_quat(cls, poses):
        """
        :param poses: (B, 69)
        """
        B, _ = poses.shape
        J = cls.NUM_BODY_JOINTS
        poses = poses.view(B, J, 3)
        poses_quat = axis_angle_to_quaternion(poses)
        assert poses_quat.shape == (B, J, 4)
        return poses_quat


SMPL_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand'
]


def batched_gradient(features):
    """
    Compute gradient of a batch of feature maps
    :param features: a 3D tensor for a batch of feature maps, dim: (N, C, H, W)
    :return: gradient maps of input features, dim: (N, 2*C, H, W), the last row and column are padded with zeros
             (N, 0:C, H, W) = dI/dx, (N, C:2C, H, W) = dI/dy
    """
    H = features.size(-2)
    W = features.size(-1)
    C = features.size(1)
    N = features.size(0)
    grad_x = (features[:, :, :, 2:] - features[:, :, :, :W - 2]) / 2.0
    grad_x = F.pad(grad_x, (1, 1, 0, 0), mode='replicate')
    grad_y = (features[:, :, 2:, :] - features[:, :, :H - 2, :]) / 2.0
    grad_y = F.pad(grad_y, (0, 0, 1, 1), mode='replicate')
    grad = torch.cat([grad_x.view(N, C, H, W), grad_y.view(N, C, H, W)], dim=1)
    return grad


class MyGridSample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grid, feat):
        vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True).detach()
        ctx.save_for_backward(feat, grid)
        return vert_feat

    @staticmethod
    def backward(ctx, grad_output):
        feat, grid = ctx.saved_tensors

        # Gradient for grid
        N, C, H, W = feat.shape
        _, Hg, Wg, _ = grid.shape
        feat_grad = batched_gradient(feat)      # dim: (N, 2*C, H, W)
        grid_grad = F.grid_sample(feat_grad, grid, mode='bilinear', padding_mode='zeros', align_corners=True)       # dim: (N, 2*C, Hg, Wg)
        grid_grad = grid_grad.view(N, 2, C, Hg, Wg).permute(0, 3, 4, 2, 1).contiguous()         # dim: (N, Hg, Wg, C, 2)
        grad_output_perm = grad_output.permute(0, 2, 3, 1).contiguous()                         # dim: (N, Hg, Wg, C)
        grid_grad = torch.bmm(grad_output_perm.view(N * Hg * Wg, 1, C),
                              grid_grad.view(N * Hg * Wg, C, 2)).view(N, Hg, Wg, 2)
        grid_grad[:, :, :, 0] = grid_grad[:, :, :, 0] * (W - 1) / 2
        grid_grad[:, :, :, 1] = grid_grad[:, :, :, 1] * (H - 1) / 2

        # Gradient for feat
        feat_d = feat.detach()
        feat_d.requires_grad = True
        grid_d = grid.detach()
        grid_d.requires_grad = True
        with torch.enable_grad():
            vert_feat = F.grid_sample(feat_d, grid_d, mode='bilinear', padding_mode='zeros', align_corners=True)
            vert_feat.backward(grad_output.detach())
        feat_grad = feat_d.grad

        return grid_grad, feat_grad
