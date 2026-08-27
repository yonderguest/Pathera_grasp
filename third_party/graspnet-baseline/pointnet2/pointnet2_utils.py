"""CPU-only replacement for GraspNet's pointnet2 CUDA helpers.

This module reproduces the inference-time primitives used by
``pointnet2_modules.py`` and ``models.modules.py`` with vectorised PyTorch
operations.  It does not implement custom gradients, which is fine for the
``model.eval()`` path used by the grasp candidate provider.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import pytorch_utils as pt_utils


def _index_by_gather(features, indices):
    """Gather feature columns according to an integer index tensor.

    ``features`` has shape ``(B, C, N)``.  ``indices`` may have shape
    ``(B, M)`` or ``(B, M, K)``; the returned tensor keeps the trailing shape.
    """
    B, C, N = features.shape
    flat = indices.reshape(B, -1)
    flat = flat.unsqueeze(1).expand(B, C, flat.size(1))
    gathered = features.gather(2, flat)
    return gathered.view(B, C, *indices.shape[1:])


def furthest_point_sample(xyz, npoint):
    """Greedy furthest point sampling on CPU."""
    xyz = xyz.contiguous()
    B, N, _ = xyz.shape
    device = xyz.device
    npoint = int(npoint)
    if npoint >= N:
        return torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)

    indices = torch.zeros((B, npoint), dtype=torch.long, device=device)
    distance = torch.full((B, N), float("inf"), dtype=xyz.dtype, device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(npoint):
        indices[:, i] = farthest
        centroid = xyz.gather(
            1, farthest.view(B, 1, 1).expand(B, 1, 3)
        ).view(B, 3)
        dist = torch.sum((xyz - centroid.unsqueeze(1)) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=1).indices
    return indices


def gather_operation(features, idx):
    return _index_by_gather(features.contiguous(), idx)


def grouping_operation(features, idx):
    return _index_by_gather(features.contiguous(), idx)


def ball_query(radius, nsample, xyz, new_xyz):
    """Return the ``nsample`` nearest points within ``radius``."""
    radius = float(radius)
    nsample = int(nsample)
    xyz = xyz.contiguous()
    new_xyz = new_xyz.contiguous()
    dist = torch.cdist(new_xyz, xyz)
    valid_dist = torch.where(
        dist < radius,
        dist,
        torch.full_like(dist, float("inf")),
    )
    top = torch.topk(valid_dist, k=min(nsample, xyz.size(1)), dim=-1, largest=False)
    indices = top.indices
    B, M, K = indices.shape
    if K < nsample:
        pad = indices[:, :, :1].expand(B, M, nsample - K)
        indices = torch.cat([indices, pad], dim=-1)
    return indices


def cylinder_query(radius, hmin, hmax, nsample, xyz, new_xyz, rot):
    """Return the ``nsample`` nearest points inside a local cylinder."""
    radius = float(radius)
    hmin = float(hmin)
    hmax = float(hmax)
    nsample = int(nsample)

    xyz = xyz.contiguous()
    new_xyz = new_xyz.contiguous()
    rot = rot.reshape(rot.size(0), rot.size(1), 3, 3).contiguous()

    # delta: (B, N, M, 3)
    delta = xyz.unsqueeze(2) - new_xyz.unsqueeze(1)
    # local: delta @ R  ->  (B, N, M, 3)
    local = torch.einsum("bnmc,bmcd->bnmd", delta, rot)
    radial_sq = local[..., 1] ** 2 + local[..., 2] ** 2
    height = local[..., 0]
    valid = (radial_sq < radius * radius) & (height > hmin) & (height < hmax)

    # Move M before N for topk over points.
    radial_sq = radial_sq.permute(0, 2, 1)
    valid = valid.permute(0, 2, 1)
    valid_dist = torch.where(
        valid,
        radial_sq,
        torch.full_like(radial_sq, float("inf")),
    )
    top = torch.topk(
        valid_dist,
        k=min(nsample, xyz.size(1)),
        dim=-1,
        largest=False,
    )
    indices = top.indices
    B, M, K = indices.shape
    if K < nsample:
        pad = indices[:, :, :1].expand(B, M, nsample - K)
        indices = torch.cat([indices, pad], dim=-1)
    return indices


def three_nn(unknown, known):
    unknown = unknown.contiguous()
    known = known.contiguous()
    dist = torch.cdist(unknown, known)
    top = torch.topk(dist, k=3, dim=-1, largest=False)
    return top.values, top.indices


def three_interpolate(features, idx, weight):
    features = features.contiguous()
    B, C, _ = features.shape
    gathered = _index_by_gather(features, idx)
    weighted = gathered * weight.unsqueeze(1)
    return weighted.sum(dim=-1)


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(
            X, theta, self.train, self.inplace
        )


class QueryAndGroup(nn.Module):
    def __init__(
        self,
        radius,
        nsample,
        use_xyz=True,
        ret_grouped_xyz=False,
        normalize_xyz=False,
        sample_uniformly=False,
        ret_unique_cnt=False,
    ):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt

    def forward(self, xyz, new_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = grouping_operation(xyz.transpose(1, 2).contiguous(), idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz = grouped_xyz / self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            new_features = (
                torch.cat([grouped_xyz, grouped_features], dim=1)
                if self.use_xyz
                else grouped_features
            )
        else:
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            unique_cnt = torch.full(
                idx.shape[:2], 0, dtype=torch.float32, device=idx.device
            )
            ret.append(unique_cnt)
        return ret[0] if len(ret) == 1 else tuple(ret)


class GroupAll(nn.Module):
    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        super().__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz

    def forward(self, xyz, new_xyz, features=None):
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            new_features = (
                torch.cat([grouped_xyz, grouped_features], dim=1)
                if self.use_xyz
                else grouped_features
            )
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        return new_features


class CylinderQueryAndGroup(nn.Module):
    def __init__(
        self,
        radius,
        hmin,
        hmax,
        nsample,
        use_xyz=True,
        ret_grouped_xyz=False,
        normalize_xyz=False,
        rotate_xyz=True,
        sample_uniformly=False,
        ret_unique_cnt=False,
    ):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.hmin = hmin
        self.hmax = hmax
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.rotate_xyz = rotate_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt

    def forward(self, xyz, new_xyz, rot, features=None):
        B, npoint, _ = new_xyz.shape
        idx = cylinder_query(
            self.radius,
            self.hmin,
            self.hmax,
            self.nsample,
            xyz,
            new_xyz,
            rot.view(B, npoint, 9),
        )
        grouped_xyz = grouping_operation(xyz.transpose(1, 2).contiguous(), idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz = grouped_xyz / self.radius
        if self.rotate_xyz:
            grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous()
            grouped_xyz_ = torch.matmul(grouped_xyz_, rot)
            grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            new_features = (
                torch.cat([grouped_xyz, grouped_features], dim=1)
                if self.use_xyz
                else grouped_features
            )
        else:
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            unique_cnt = torch.full(
                idx.shape[:2], 0, dtype=torch.float32, device=idx.device
            )
            ret.append(unique_cnt)
        return ret[0] if len(ret) == 1 else tuple(ret)

