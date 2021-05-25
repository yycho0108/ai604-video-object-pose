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

def create_corners(dimension, location=None, R=None):
    dx = dimension[0] / 2
    dy = dimension[1] / 2
    dz = dimension[2] / 2

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                x_corners.append(dx*i)
                y_corners.append(dy*j)
                z_corners.append(dz*k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])

    return final_corners


# FIXME(Jiyong): debug
def calc_location(box_2d, intrinsic_matrix, dimension, quaternion):
    #global orientation
    R = quaternion_to_matrix(th.as_tensor(quaternion)).cpu().numpy()

    # format 2d corners
    # box_2d is (top, left, height, width)
    xmin = box_2d[1]
    xmax = box_2d[3] + xmin
    ymin = box_2d[0]
    ymax = box_2d[2] + ymin

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]

    # # get the point constraints
    # constraints = []

    # left_constraints = []
    # right_constraints = []
    # top_constraints = []
    # bottom_constraints = []

    # using a different coord system
    dx = dimension[0] / 2
    dy = dimension[1] / 2
    dz = dimension[2] / 2

    vertices = []
    for i in (-1,1):
        for j in (-1,1):
            for k in (-1,1):
                vertices.append([i*dx, j*dy, k*dz])
    
    constraints = list(itertools.permutations(vertices, 4))
    print(len(constraints))

    # # left and right of object
    # for i in (-1,1):
    #     left_constraints.append([dx, i*dy, -dz])
    # for i in (-1,1):
    #     right_constraints.append([dx, i*dy, dz])

    # # top and bottom of object
    # for i in (-1,1):
    #     for j in (-1,1):
    #         top_constraints.append([i*dx, -dy, j*dz])
    # for i in (-1,1):
    #     for j in (-1,1):
    #         bottom_constraints.append([i*dx, dy, j*dz])

    # # now, 64 combinations
    # for left in left_constraints:
    #     for top in top_constraints:
    #         for right in right_constraints:
    #             for bottom in bottom_constraints:
    #                 constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    # constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [np.inf]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = np.zeros([4,3], dtype=np.float)
        b = np.zeros([4,1])

        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = np.dot(R, X)
            M[:3,3] = RX.reshape(3)

            M = np.dot(intrinsic_matrix, M)

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[index,3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

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

    # def project_3d_pt(pt, cam_to_img):
    #     point = np.array(pt)
    #     point = np.append(point, 1)
    #     point = np.dot(cam_to_img, point)
    #     print("Projected point:", point)
    #     point = point[:2]/point[2]
    #     point = point * [h, w]
    #     point = point.astype(np.int16)
    #     return point
    
    # corners = create_corners(dimension, location=center, R=rotation)

    # box_3d = []
    # for corner in corners:
    #     print("corner:", corner)
    #     point = project_3d_pt(corner, cam_to_img)
    #     print("point:", point)
    #     box_3d.append(point)

    # # box_3d = pt[1:]
    # box_3d = np.asarray(box_3d, dtype=np.int16)

    # TODO(Jiyong): put into loop
    print(img)
    print(img.shape)
    print(type(img))
    print(img.dtype)
    print(box_3d.dtype)
    print(box_3d[0][0])

    # CHW -> HWC
    if isinstance(img, th.Tensor):
        img = img.detach().cpu().numpy()
    img = np.ascontiguousarray(img.transpose(1,2,0))

    # draw points
    for kp in box_3d:
        cv2.circle(img, (kp[0], kp[1]), 3, cv_colors.RED.value, 3)
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

def plot_regressed_3d_bbox(img, box_2d, proj_matrix, dimension, quaternion, translations=None):
    location, X = calc_location(box_2d, proj_matrix, dimension, quaternion)
    rotation = quaternion_to_matrix(th.as_tensor(quaternion)).cpu().numpy()
    # # c, h, w = img.shape
    # # img = np.array(img/255., dtype=np.float32)
    # # img = img.transpose(1,2,0) # (h,w,c)
    # # img = img.reshape(h,w,c) # (h,w,c)
    # # assert img.flags['C_CONTIGUOUS'] == True
    # # img = Image.fromarray(img.astype(np.uint8))
    if translations is not None:
        location = translations
    # else:
    #     img = plot_3d_box(img, proj_matrix, rotation, dimension, location)
    # # img = np.array(img*255., dtype=np.int)
    # # img = img.transpose(2,0,1) # (c,h,w)
    # # img = img.reshape(c,h,w)

    img = plot_3d_box(img, proj_matrix, rotation, dimension, location)
    return img
