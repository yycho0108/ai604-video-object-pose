"""Solving for 3D Box Translation
Reference: https://cs.gmu.edu/~amousavi/papers/3D-Deepbox-Supplementary.pdf, http://ywpkwon.github.io/pdf/bbox3d-study.pdf
Code Reference: https://github.com/skhadem/3D-BoundingBox/blob/master/
"""

from enum import Enum
import itertools
import numpy as np
import cv2
import torch as th
from simple_parsing.helpers.serialization.serializable import D
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from torch._C import dtype


class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)

<<<<<<< HEAD

=======
def get_cube_points() -> th.Tensor:
    """ Get cube points, sorted in descending order by axes and coordinates. """
    points_3d = list(itertools.product(
        *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))
#     points_3d = np.insert(points_3d, 0, [0, 0, 0], axis=0)
    points_3d = th.as_tensor(points_3d, dtype=th.float32).reshape(-1, 3)
    return points_3d

def calc_location(box_2d, proj_matrix, dimension, quaternion, translations):
>>>>>>> f800453d7f95b887c3b126c08b9b8d1750802a4b
    #global orientation
    R = (quaternion_to_matrix(th.as_tensor(quaternion))
         .detach().cpu().numpy())

    # format 2d corners
<<<<<<< HEAD

=======
#     xmin, ymin, xmax, ymax = box_2d

    # left top right bottom
#     box_corners = [xmin, ymin, xmax, ymax]
    box_corners = box_2d
    
    T_box = np.zeros((4, 4), dtype=np.float32)
    T_box[..., :3, :3] = np.matmul(R, np.diag(dimension))
    T_box[..., :3, -1] = translations
    T_box[..., 3, 3] = 1
    T_p = proj_matrix.reshape(4, 4)
    T = np.matmul(T_p, T_box)
    T = np.array(T)
    
    v3 = np.einsum('...ab, kb -> ...ka',
                      T[..., :3, :3], get_cube_points()) + T[..., None, :3, -1]
    v2 = v3[...,:2] / v3[...,2:]
    
    imin = np.argmin(v2, 0)
    imax = np.argmax(v2, 0)
    indices = [imin[0], imin[1], imax[0], imax[1]]
    vertices = get_cube_points() * dimension
    constraints = [vertices[indices]]
#     constraints = list(itertools.permutations(vertices, 4))
>>>>>>> f800453d7f95b887c3b126c08b9b8d1750802a4b

    best_loc = None
    best_error = [np.inf]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    count = 0
    bs = np.inf
    K = proj_matrix
    for XX in constraints:        
        # create A, b
        A = np.zeros([4,3], dtype=np.float32)
        b = np.zeros([4,1], dtype=np.float32)
        for row, a in enumerate([0,1,0,1]):
            v = box_corners[row]
            A[row, :] = v * K[2, :3] - K[a, :3]
            b[row, :] = (-v * np.einsum('a,b,ba->', XX[row], K[2, :3], R)
                       + np.einsum('a,b,ba->', XX[row], K[a, :3], R))


        # solve here with least squares, since over fit will get some error
#         print('A', A,  'b', b)
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # found a better estimation
        if error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = XX

    # return best_loc, [left_constraints, right_constraints] # for debugging
    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    print("lstsq error:", best_error)
    return best_loc, best_X

def plot_3d_box(img, cam_to_img, rotation, dimension, center):
    c, h, w = img.shape
    # takes in a 3d point and projects it into 2d
    vertices = list(itertools.product(*zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))  # 8x3
    vertices = np.array(vertices)

    T_box = np.zeros((4, 4))
    T_box[:3, :3] = np.matmul(rotation, np.diag(dimension))
    T_box[:3, -1] = center
    T_box[3, 3] = 1
    T = np.matmul(cam_to_img, T_box)
    T = np.array(T)

    # inner product
    box_3d = np.einsum('ab, kb -> ka', T[:3, :3], vertices) + T[None, :3, -1]
    box_3d[..., :-1] /= box_3d[..., -1:]
    # NDC(-1~1) -> UV(0~1) coordinates
    box_3d[..., :-1] = 0.5 + 0.5 * box_3d[..., :-1]
    # This flipping is technically a bug accounting for the
    # inconsistency in the keypoint convention from the objectron dataset.
    box_3d[..., :2] = np.flip(box_3d[..., :2], axis=(-1,))
    box_3d = box_3d * np.array([w, h, 1.0])
    box_3d = box_3d.astype(int)
<<<<<<< HEAD

=======
>>>>>>> f800453d7f95b887c3b126c08b9b8d1750802a4b

    # CHW -> HWC
    if isinstance(img, th.Tensor):
        img = img.detach().cpu().numpy()
    img = np.ascontiguousarray(img.transpose(1,2,0))

    # draw points
    for kp in box_3d:
        cv2.circle(img, (kp[0], kp[1]), 3, cv_colors.RED.value, 3)

    # TODO(Jiyong): put into loop
    # draw edges
    # lines along x-axis
    img = cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), cv_colors.RED.value, 1)
    img = cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.RED.value, 1)
    img = cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.RED.value, 1)
    img = cv2.line(img, (box_3d[3][0], box_3d[3][1]), (box_3d[7][0],box_3d[7][1]), cv_colors.RED.value, 1)
    # lines along y-axis
    img = cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), cv_colors.RED.value, 1)
    img = cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.RED.value, 1)
    img = cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.RED.value, 1)
    img = cv2.line(img, (box_3d[5][0], box_3d[5][1]), (box_3d[7][0],box_3d[7][1]), cv_colors.RED.value, 1)
    # lines along z-axis
    img = cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[1][0],box_3d[1][1]), cv_colors.RED.value, 1)
    img = cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.RED.value, 1)
    img = cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.RED.value, 1)
    img = cv2.line(img, (box_3d[6][0], box_3d[6][1]), (box_3d[7][0],box_3d[7][1]), cv_colors.RED.value, 1)

    # HWC -> CHW
    img = img.transpose(2,0,1)

    return img

<<<<<<< HEAD

=======
def plot_regressed_3d_bbox(img, points_2d, proj_matrix, dimension, quaternion, translations):
    # TODO(Jiyong): make with batch
    img = img[0]
    points_2d = points_2d[0]
    proj_matrix = proj_matrix[0]
    dimension = dimension[0]
    quaternion = quaternion[0]
    translations = translations[0]

    points = points_2d[..., 1:, :2]
    # Restore conventions
    points = th.flip(points, dims=(-1,)) # XY->IJ
    points = 2.0 * (points - 0.5) # (0,1) -> (-1, +1)

    pmin = points.min(dim = -2).values.reshape(-1)
    pmax = points.max(dim = -2).values.reshape(-1)
    box_2d = th.cat([pmin,pmax])

    location, X = calc_location(box_2d, proj_matrix, dimension, quaternion, translations)
    rotation = quaternion_to_matrix(th.as_tensor(quaternion)).cpu().numpy()

    img = plot_3d_box(img, proj_matrix, rotation, dimension, location)
    img = th.as_tensor(img)
>>>>>>> f800453d7f95b887c3b126c08b9b8d1750802a4b

    return img
