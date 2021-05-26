import itertools
import numpy as np
import torch as th
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix


def calc_location(box_2d, proj_matrix, dimension, quaternion, gt_trans):
    #global orientation
    R = quaternion_to_matrix(th.as_tensor(quaternion)).cpu().numpy()

    # format 2d corners
    # box_2d is (top, left, height, width)
    xmin = box_2d[1]
    xmax = (box_2d[3] + xmin)
    ymin = box_2d[0]
    ymax = (box_2d[2] + ymin)

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]

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

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [np.inf]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
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

            K = proj_matrix[:3, :]
            M = np.dot(K, M)

            # ref: http://ywpkwon.github.io/pdf/bbox3d-study.pdf
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


if __name__ == '__main__':
    # box_2d: tensor([[0.0273, 0.0623, 0.6438, 0.9307]])
    # gt_trans: tensor([[-0.3957,  0.0739, -1.8345]], device='cuda:0')
    # gt_dim: tensor([[0.5555, 0.8413, 1.2966]], device='cuda:0')
    # gt_quat: tensor([[ 0.1499, -0.4744,  0.7188,  0.4857]], device='cuda:0')
    # pj_mat: tensor([ 1.6358e+00,  0.0000e+00,  1.9637e-02,  0.0000e+00,  0.0000e+00,
    #          2.1811e+00, -3.0700e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #         -1.0000e+00, -1.0000e-03,  0.0000e+00,  0.0000e+00, -1.0000e+00,
    #          0.0000e+00], device='cuda:0')

    box_2d = np.array([0.0273, 0.0623, 0.6438, 0.9307])
    proj_matrix = np.array([1.6358e+00,  0.0000e+00,  1.9637e-02,  0.0000e+00,  
                            0.0000e+00,  2.1811e+00, -3.0700e-03,  0.0000e+00,  
                            0.0000e+00,  0.0000e+00, -1.0000e+00, -1.0000e-03,  
                            0.0000e+00,  0.0000e+00, -1.0000e+00,  0.0000e+00]).reshape(4,4)
    dimension = np.array([ 0.1499, -0.4744,  0.7188,  0.4857])
    quaternion = np.array([0.5547,  0.4986, -0.3726,  0.5521])
    translations = np.array([-0.3957,  0.0739, -1.8345])

    location, X = calc_location(box_2d, proj_matrix, dimension, quaternion, translations)

    print("cal_translations:", location)
    print("truth_translations:", translations)