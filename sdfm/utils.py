import sys
from typing import Callable, Iterable

import numpy as np


def sort_objects(objs: list, filter_func: Callable = None, filter_keys: Iterable = None):
    """Sort objects into a grouped dict according to the provided filter function."""
    assert not (filter_func is None and filter_keys is None)
    if filter_keys is not None:
        values = list(filter_keys)
        assert len(values) == len(objs)
    else:
        values = [filter_func(_) for _ in objs]
    sorted_data = {k: [] for k in set(values)}
    for val, obj in zip(values, objs):
        sorted_data[val].append(obj)
    return sorted_data


class SimplePCA:
    """Scikit-learn can suck it"""

    # TODO: some tests for this?
    def __init__(self, data: np.ndarray):
        mean = data.mean(axis=0)
        cov = (_ := data - mean).T @ _
        components = np.linalg.svd(cov)[2]
        self.data, self.mean, self.components = data.copy(), mean, components

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) @ self.components.T

    def reverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data @ self.components + self.mean


def match_point_clouds(pts1: np.ndarray, pts2: np.ndarray) -> (callable, dict):
    """"""
    # TODO: this is not good enough
    transform = match_point_clouds_simple(pts1, pts2)
    vecs = pts1 - (pts2_new := transform(pts2))
    if (vecs ** 2).max() < 0.01:
        return transform, {'pts': pts2_new, 'vecs': vecs}

    # poor match -> assume point ordering is not consistent
    pca1 = SimplePCA(pts1)
    pca2 = SimplePCA(pts2)
    transform = lambda pts: pca1.reverse_transform(pca2.transform(pts))
    ordering = np.linalg.norm(pts1[:, None] - (pts2_new := transform(pts2)), axis=-1).argmin(axis=-1)
    vecs = pts1 - pts2_new[ordering]
    if (vecs ** 2).max() < 0.01:
        return transform, {'pts': pts2_new, 'vecs': vecs}

    # for spherical clouds, PCA does not find preferential directions -> try again
    transform = match_point_clouds_simple(pts1, pts2[ordering])
    vecs = pts1 - (pts2_new := transform(pts2[ordering]))
    if (vecs ** 2).max() < 0.01:
        return transform, {'pts': pts2_new, 'vecs': vecs}

    print(f'Could not match point clouds..', file=sys.stderr)


def match_point_clouds_simple(pts1: np.ndarray, pts2: np.ndarray) -> callable:
    """
    Find a transformation from pts2 to pts1. Adapted from Arun et al., 1987.
    https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence
    """
    com1, com2 = pts1.mean(axis=0), pts2.mean(axis=0)
    cov = (pts2 - com2).T @ (pts1 - com1)
    svd = np.linalg.svd(cov)
    rot = svd[2].T @ svd[0].T
    # assert np.allclose(np.linalg.det(rot), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."
    return lambda x: x @ rot.T + (com1 - rot @ com2)


def fit_plane(pts: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Fit a plane to a point cloud and return its centroid and normal vector.
    STOLEN FROM https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    """
    pts = pts.squeeze()
    assert pts.shape[1] <= pts.shape[0], f'Found only {pts.shape[0]} points for {pts.shape[1]} dimensions..'
    ctr = pts.mean(axis=0)
    cov = np.cov((pts - ctr).T)
    return ctr, np.linalg.svd(cov)[0][:, -1]


def compute_inertia(pts: np.ndarray) -> float:
    """Some measure of the size of a point cloud"""
    mean = pts.mean(axis=0)
    return np.linalg.norm(pts - mean)
