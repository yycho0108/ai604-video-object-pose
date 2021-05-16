"""Solving for 3D Box Translation
Reference: https://cs.gmu.edu/~amousavi/papers/3D-Deepbox-Supplementary.pdf, http://ywpkwon.github.io/pdf/bbox3d-study.pdf
Code Reference: https://github.com/skhadem/3D-BoundingBox/blob/master/library/Math.py
"""

import numpy as np
from pytorch3d.transforms import quaternion_to_matrix


def calc_location(box_2d, proj_matrix, dimension, quaternion):
    #global orientation
    R = quaternion_to_matrix(quaternion)

    # format 2d corners
    # box_2d is [(xmin, ymin), (xmax, ymax)]
    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]

    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    # left and right could either be the front of the object ot the back of the object
    # careful to use left and right based on image, no of actual object's left and right
    for i in (-1,1):
        left_constraints.append([dx, i*dy, -dz])
    for i in (-1,1):
        right_constraints.append([dx, i*dy, dz])

    # top and bottom are easy, just the top and bottom of object
    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy, j*dz])
    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, dy, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
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

            M = np.dot(proj_matrix, M)

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
    return best_loc, best_X