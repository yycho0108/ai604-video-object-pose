#!/usr/bin/env python3

__all__ = ['SolveTranslation']

import torch as th
import numpy as np
import itertools
import warnings
from typing import Dict, Hashable

from pytorch3d.transforms import quaternion_to_matrix
from top.data.schema import Schema

from top.data.transforms.keypoint import (
    BoxPoints2D
)

from scipy.spatial import ConvexHull
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning


def get_cube_points(prepend_centroid: bool = False) -> th.Tensor:
    """Get cube points, sorted in ascending order by axes and coordinates."""
    points_3d = list(itertools.product(
        *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))
    if prepend_centroid:
        points_3d = np.insert(points_3d, 0, [0, 0, 0], axis=0)
    points_3d = th.as_tensor(points_3d, dtype=th.float32).reshape(-1, 3)
    return points_3d


def compute_bounds(A: np.ndarray, b: np.ndarray):
    """Compute bounds along N line segments that form Ax<=b.

    Intended for feasibility-checking for 2D linear inequalities. given
    A=(N,2) and b=(N), computes the upper and lower bounds of the
    inequality along each line segment with O(N^2) time complexity.
    """
    A0, b0 = A[:, None], b[:, None]  # N,1,2
    A1, b1 = A[None, :], b[None, :]  # 1,N,2

    x0, y0 = A0[..., 0], A0[..., 1]
    x1, y1 = A1[..., 0], A1[..., 1]
    denom = x0 * y1 - y0 * x1

    cx = x0 * b1 - x1 * b0
    cy = y0 * b1 - y1 * b0

    # Compute distance along tangent.
    with np.errstate(invalid='ignore'):
        dt = -(x0 * cx + y0 * cy) / denom

    # Compute bounds.
    lo = np.max(dt, axis=-1, where=(denom > 0), initial=-np.inf)
    hi = np.min(dt, axis=-1, where=(denom < 0), initial=np.inf)
    return (lo, hi)


def _is_linprog_feasible(A_ub: np.ndarray, b_ub: np.ndarray, *ars, **kwargs):
    """Checks if a system of 2D linear inequalities is feasible.

    Emulates the arguments for `_is_linprog_feasible_scipy`.
    """
    A_ub = np.asarray(A_ub)
    b_ub = np.asarray(b_ub)

    lo, hi = compute_bounds(A_ub, b_ub)
    return (lo <= hi).any()  # , None


def _is_linprog_feasible_scipy(*args, **kwds):
    """Checks if a system of any linear inequalities is feasible.

    See scipy.optimize.linprog for arguments.
    """
    try:
        sol = linprog(*args, **kwds)
        return sol.status != 2
    except ValueError as e:
        # see line 1323 of _linprog_util.py
        return True


def compute_feasible_permutations(points, fovs,
                                  debug_chull: bool = False):
    """Compute valid permutations for 2D bounding box vertex correspondences.

    Solves for the set of possible (i_min,i_max,j_min,j_max)
    permutations given rotated points in camera coordinates (with
    Objectron convention) and the camera intrinsic parameters (fov,
    required for frustum ray unprojection).

    Generally reduces the set of possible matches from 4096 to 3~21 options.
    """
    perms = []

    # NOTE(ycho): this could certainly be better written, but works for now.
    # we're essentially looping over modified coordinates in
    # [data] (-Z,X),(-Z,Y) --> [algo] (Z,Y), (Z,X) order
    # to match the convention I was using when I was drafting the algorithm.
    # That is, +Z forward, +X rightward, +Y downward (ROS optical frame convention)
    # rather than +Z backward, +Y rightward, +X downward (Objectron convention)
    views = [((2, 0), (-1, 1), 1), ((2, 1), (-1, 1), 0)]

    for i, (axs, sgn, ax) in enumerate(views):
        # Determine the rays defining the frustum boundaries.
        fov = fovs[axs[1]]
        kmax = np.asarray([-np.sin(fov / 2), np.cos(fov / 2)])
        kmin = np.asarray([np.sin(fov / 2), np.cos(fov / 2)])

        # Project `points` down to current plane.
        xax = points[..., axs] * sgn

        if debug_chull:
            # NOTE(ycho): Explicit convex-hull check.
            # Guaranteed CCW.
            hull = ConvexHull(xax)
            # Make into CW, since that's how I thought about this
            indices = hull.vertices[::-1]
            x = xax[indices]
        else:
            # NOTE(ycho): Simpler routine to compute convex hull,
            # by exploiting cuboid geometry.
            i0 = np.argmin(points[..., ax])
            i1 = (7 - i0) % 8  # == np.argmin(points[...,ax])
            indices = np.arange(8)
            indices = np.delete(indices, [i0, i1], axis=0)

            # NOTE(ycho): Sort vertices of convex hull, CW.
            angles = np.arctan2(xax[indices, 0], xax[indices, 1])
            indices = indices[np.argsort(angles)]
            x = xax[indices]

        # Inequalities by camera fov.
        # has to be higher than dmax
        dmax = np.max(x.dot(kmax))
        # has to be smalller than dmin
        dmin = np.min(x.dot(kmin))

        # u = Unit vectors around hull periphery
        # such that u[i] = normalized(x[i] - x[i-1])
        # v = u.cross(\hat{z}), orthogonal vectors
        prv = np.roll(x, 1, axis=0)
        u = x - prv
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        v = np.stack([-u[:, 1], u[:, 0]], axis=-1)

        # Check for max
        i_minmax = []
        c_dummy = np.zeros(2)
        for i in range(len(v)):
            i_nxt = (i + 1) % len(v)

            # Check feasibility on max
            A_ub_max = [-kmax, kmin, -v[i], v[i_nxt]]
            b_ub_max = [-dmax, dmin, -v[i].dot(x[i]), v[i_nxt].dot(x[i_nxt])]
            A_ub_max = np.asarray(A_ub_max, dtype=np.float32)
            b_ub_max = np.asarray(b_ub_max, dtype=np.float32)

            # A_ub @ x <= b_ub
            max_ok = _is_linprog_feasible(
                c=c_dummy,
                A_ub=A_ub_max,
                b_ub=b_ub_max,
                bounds=(None, None)
            )

            if max_ok:
                # Check feasibility on min
                for j in range(len(v)):
                    # Can't be both min&max at the same time
                    if i == j:
                        continue
                    j_nxt = (j + 1) % len(v)

                    # Check feasibility on max+min
                    A_ub = np.concatenate(
                        [A_ub_max, [v[j], -v[j_nxt]]], axis=0)
                    b_ub = np.concatenate(
                        [b_ub_max, [v[j].dot(x[j]), -v[i_nxt].dot(x[i_nxt])]], axis=0)

                    # A_ub @ x <= b_ub
                    min_ok = _is_linprog_feasible(
                        c=c_dummy, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
                    if min_ok:
                        # NOTE(ycho): j=min, i=max
                        i_minmax.append((indices[j], indices[i]))
        perms.append(i_minmax)
    return ((i0, j0, i1, j1)
            for (i0, i1), (j0, j1) in itertools.product(*perms))


class SolveTranslation:
    """Solve the translational component of the 3d oriented bounding cuboid.

    Based on the detected (scale, orientation), compute the remaining
    translational DoF from the constraints obtained by the 2D bounding
    box detections; internally tries all possible vertex permutations
    and chooses the best one that minimizes the error.
    """

    def __init__(self, recompute_error: bool = True,
                 debug_chull: bool = False):
        self.recompute_error = recompute_error
        self.debug_chull = debug_chull
        self.points = get_cube_points()
        self.box_points = BoxPoints2D(th.device('cpu'),
                                      key_out=Schema.KEYPOINT_2D)

    def __call__(self, inputs: Dict[Hashable,
                                    th.Tensor]) -> Dict[Hashable, th.Tensor]:
        proj_matrix = inputs[Schema.PROJECTION]
        # NOTE(ycho): BOX_2D = (i0, j0, i1, j1) in normalized coords (-1,1).
        box_2d = inputs[Schema.BOX_2D]

        # NOTE(ycho): outputs from network
        dimension = inputs[Schema.SCALE]
        if Schema.ORIENTATION in inputs:
            R = inputs[Schema.ORIENTATION].detach().cpu().numpy()
        elif Schema.QUATERNION in inputs:
            quaternion = inputs[Schema.QUATERNION]
            R = (quaternion_to_matrix(th.as_tensor(quaternion))
                 .detach().cpu().numpy())
        else:
            raise KeyError('Orientation information Not Found!')

        vertices = (self.points.cpu() * dimension.cpu()).detach().numpy()

        if True:
            # Reduce the number of permutations through geometric reasoning.
            fovs = 2.0 * np.arctan(1.0 / proj_matrix[[0, 1], [0, 1]])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore', category=LinAlgWarning)
                warnings.filterwarnings(
                    action='ignore', category=OptimizeWarning)
                warnings.filterwarnings(
                    action='ignore',
                    category=np.VisibleDeprecationWarning)
                warnings.filterwarnings(
                    action='ignore',
                    category=RuntimeWarning)
                perms = compute_feasible_permutations(
                    vertices @ R.T, fovs, self.debug_chull)
            perms = np.asarray(list(perms), dtype=np.int32)
            constraints = vertices[perms, :]
        else:
            constraints = list(itertools.permutations(vertices, 4))

        # Initialize current best candidates.
        best_loc = None
        best_error = np.inf
        best_X = None

        # Loop through each possible constraint, hold on to the best guess
        K = proj_matrix.detach().cpu().numpy()

        # Create design matrices Ax=b for SVD.
        # K_ax is the axes of K repeated for each corresponding spatial axis.
        K_ax = K[(0, 1, 0, 1), :3]
        # TODO(ycho): use integer permutations directly and index into the array,
        # instead of creating (large) redundant copies
        for X in constraints:
            A = np.einsum('n,a->na', box_2d, K[2, :3]) - K_ax
            b = -np.einsum('na,ab,nb->n', A, R, X)

            # Solve here with least squares since overparameterized.
            # NOTE(ycho): `error` here indicates algebraic error;
            # it's generally preferable to use geometric error.
            loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

            # Evaluate solution ...
            if self.recompute_error:
                # NOTE(ycho): evaluate error based on match with box.
                # FIXME(ycho): This probably results in much more expensive
                # evaluation.
                args = {
                    Schema.ORIENTATION: th.as_tensor(R).detach().cpu(),
                    Schema.TRANSLATION: th.as_tensor(loc).detach().cpu(),
                    Schema.SCALE: th.as_tensor(dimension).detach().cpu(),
                    Schema.PROJECTION: th.as_tensor(K).detach().cpu(),
                    Schema.INSTANCE_NUM: 1
                }
                out_points = self.box_points(args)[Schema.KEYPOINT_2D][..., :2]
                out_points = th.flip(out_points, dims=(-1,))  # XY->IJ
                out_points = 2.0 * (out_points - 0.5)  # (0,1) -> (-1, +1)
                pmin = out_points.min(dim=-2).values.reshape(-1)
                pmax = out_points.max(dim=-2).values.reshape(-1)
                out_box_2d = th.cat([pmin, pmax])
                error2 = th.norm(box_2d - out_box_2d.to(box_2d.device))
                error = error2

            # Update estimate with better alternative.
            if (error < best_error):
                best_loc = loc
                best_error = error
                best_X = X

        return best_loc, best_X
