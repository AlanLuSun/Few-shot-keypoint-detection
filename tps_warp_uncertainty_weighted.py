import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import itertools

class TPS:
    @staticmethod
    def tps_theta_from_points(src_pts, dst_pts, lambd=0., uncertainty_square=None):

        n = src_pts.shape[0]

        R = TPS.u(TPS.d(src_pts, src_pts))
        if uncertainty_square is None:
            K = R + np.eye(n, dtype=np.float32) * lambd  # R + lambda * I
        else:
            assert len(uncertainty_square) == n  # a vector has length of n
            K = R + np.diag(uncertainty_square).astype(np.float32) * lambd  # R + lambda * D^{-2}

        P = np.ones((n, 3), dtype=np.float32)  # homogeneous src kps, n x 3, [[1, x1, y1], [1, x2, y2], ...]
        P[:, 1:] = src_pts

        V = np.zeros((n + 3, 2), dtype=np.float32)  # (n+3) x 2
        V[:n, :] = dst_pts

        L = np.zeros((n + 3, n + 3), dtype=np.float32)
        L[:n, :n] = K
        L[:n, -3:] = P
        L[-3:, :n] = P.T

        try:
            theta = np.linalg.solve(L, V)  # n x 2, theta has structure [W^T, A^T]
        except:
            print('singular matrix error.', src_pts, dst_pts)
            theta = None

        return theta

    @staticmethod
    def d(a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r ** 2 * np.log(r + 1e-9)

    @staticmethod
    def apply_transform(input_pts, src_pts, theta):
        '''
        :param input_pts: K x 2
        :param src_pts: N x 2, control points
        :param theta: TPS params solved from corresponding points, (N+3) x 2
        :return: warped pts
        '''
        k = input_pts.shape[0]
        R = TPS.u(TPS.d(input_pts, src_pts))  # K x N
        R2 = np.column_stack((R, np.ones(k), input_pts))  # K x (N + 3)
        # W, A = theta[:-3], theta[-3:]
        warped_pts = np.matmul(R2, theta)  # K x 2

        return warped_pts

def uniform_grid(shape, unit_space=True):
    '''Uniform grid coordinates.

    Params
    ------
    shape : tuple
        H x W defining the number of height and width dimension of the grid

    Returns
    -------
    points: HxWx2 tensor
        Grid coordinates over [0,1] normalized image range. Each position's entry is (x, y)
    '''

    H, W = shape[:2]
    grids = np.empty((H, W, 2))
    if unit_space == True:
        grids[..., 0] = np.linspace(0, 1, W, dtype=np.float32)
        grids[..., 1] = np.expand_dims(np.linspace(0, 1, H, dtype=np.float32), -1)
    else:
        grids[..., 0] = np.linspace(0, W-1, W, dtype=np.float32)
        grids[..., 1] = np.expand_dims(np.linspace(0, H-1, H, dtype=np.float32), -1)

    return grids

def uniform_grid2(shape):
    '''
    :param shape: (H, W)
    :return: return an generator, of which each entry is (y, x), in total of (H x W) entries. Example:

        import itertools
        target_height, target_width = 3, 4
        a = itertools.product(range(target_height), range(target_width))
        print(list(a))
        # output: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    '''
    height, width = shape[:2]
    grids = itertools.product(range(height), range(width))
    return grids


def tps_grid(theta, src_pts, dst_shape, unit_space=True):
    '''
    warp a set of grids
    :param theta: (N + 3) x 2
    :param src_points: N x 2, namely the control points
    :param dshape: (H', W'), dst shape
    :param unit_space: if unit_space is true, the src_pts should be in range [0, 1]
    :return: warped grids with shape of (H' x W') x 2
    '''
    ugrids = uniform_grid(dst_shape, unit_space=unit_space)  # H x W x 2, each element is (x, y)
    ugrids = ugrids.reshape(-1, 2)  # (H*W) x 2, each element per row is (x, y)
    dgrids = TPS.apply_transform(ugrids, src_pts, theta)  # (H*W) x 2
    # dgrids = dgrids.reshape(*dst_shape, 2)  # (H x W) x 2

    return dgrids, ugrids  # (H' x W') x 2, grid[i,j] in range [0..1] or in its original space


def tps_grid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.

    Params
    ------
    grid : H x W x 2 array
        Normalized flow field coordinates as computed by computed dense grid.
    sshape : tuple
        Height and width of source image in pixels.

    Returns
    -------
    mapx : H x W array
    mapy : H x W array
    '''
    H, W = sshape[:2]
    mx = (grid[:, :, 0] * W).astype(np.float32)
    my = (grid[:, :, 1] * H).astype(np.float32)

    return mx, my

def warp_image(img, src_pts, dst_pts, dshape=None, unit_space=True, lambd=0, uncertainty_square=None, method_for_new_pts=0):
    dshape = dshape or img.shape[:2]
    theta = TPS.tps_theta_from_points(dst_pts, src_pts, lambd=lambd, uncertainty_square=uncertainty_square)
    if theta is None:  # singular matrix error occurs
        return None, None
    grid, ugrid = tps_grid(theta, dst_pts, dshape, unit_space=unit_space)  # tps_grids and uniform grids
    grid_2D = grid.reshape(*dshape, 2)  # H x W x 2
    if unit_space == True:
        mapx, mapy = tps_grid_to_remap(grid_2D, img.shape[:2])  # H x W, H x W
    else:
        mapx, mapy = grid_2D[:,:, 0].astype(np.float32), grid_2D[:,:, 1].astype(np.float32)
    warped_img = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

    if method_for_new_pts == 0:
        d_matrix = TPS.d(src_pts, grid)
        ind = np.argmin(d_matrix, axis=1)
        d = d_matrix[range(0, len(ind)), ind]  # index the value: d_matrix[0, ind[0]], ..., d_matrix[k, ind[k]]
        mask = d <= 5
        new_src_pts = np.zeros((src_pts.shape[0], 2)) -1  # initialize to be (-1, -1) for out-of-dshape new pts
        new_src_pts[mask] = ugrid[ind[mask]]
    else:
        theta2 = TPS.tps_theta_from_points(src_pts, dst_pts, lambd=lambd, uncertainty_square=uncertainty_square)
        new_src_pts = TPS.apply_transform(src_pts, src_pts, theta2)

    return warped_img, new_src_pts

def show_warped(img, warped, src_pts, dst_pts, unit_space=True, new_src_pts=None):
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(img[...,::-1], origin='upper')
    axs[1].imshow(warped[..., ::-1], origin='upper')
    if unit_space:
        axs[0].scatter(src_pts[:, 0]*img.shape[1], src_pts[:, 1]*img.shape[0], marker='+', color='red', s=80, linewidth=2)
        axs[0].scatter(dst_pts[:, 0]*img.shape[1], dst_pts[:, 1]*img.shape[0], marker='o', color='', edgecolor='blue', s=80, linewidth=2)
        axs[1].scatter(dst_pts[:, 0]*warped.shape[1], dst_pts[:, 1]*warped.shape[0], marker='o', color='', edgecolor='blue', s=80, linewidth=2)
        if new_src_pts is not None:
            axs[1].scatter(dst_pts[:, 0] * warped.shape[1], dst_pts[:, 1] * warped.shape[0], marker='+', color='red', s=80, linewidth=2)
    else:
        axs[0].scatter(src_pts[:, 0] , src_pts[:, 1], marker='+', color='red', s=80, linewidth=2)
        axs[0].scatter(dst_pts[:, 0] , dst_pts[:, 1], marker='o', color='', edgecolor='blue', s=80, linewidth=2)
        axs[1].scatter(dst_pts[:, 0] , dst_pts[:, 1], marker='o', color='', edgecolor='blue', s=80, linewidth=2)
        if new_src_pts is not None:
            axs[1].scatter(new_src_pts[:, 0], new_src_pts[:, 1], marker='+', color='red', s=80, linewidth=2)

    plt.show()

def preprocess_uncertainty(src_pts, dst_pts):
    d = np.sqrt(((dst_pts - src_pts) ** 2).sum(-1))  # N
    d_max = np.max(d)
    d_mean = np.mean(d)
    d_med = np.median(d)
    d_p = np.percentile(d, 80, axis=0)
    l = 0.5 * d_max
    lambd = TPS.u(l)
    beta = 1
    uncertainty = np.array([1, 5, 1, 1], dtype=np.float32)
    print('lambd: %f, beta: %f'%(lambd, beta))

    print(uncertainty)
    D = 1 / uncertainty
    print(D)
    D = D / sum(D)
    D = np.power(D, beta)
    D = D / sum(D)
    D = 1 / D
    print(D)

    # s = sum(uncertainty)
    # D = uncertainty / s
    # D = np.power(D, beta)
    # D = D / sum(D)
    # D = s * D
    # print(D)

    return lambd, D

if __name__=='__main__':
    img = cv2.imread('image.png')

    c_src = np.array([
        [0.0, 0.0],
        [1., 0],
        [1, 1],
        [0, 1],
        [0.3, 0.3],
        [0.7, 0.7],
    ])

    c_dst = np.array([
        [0., 0],
        [1., 0],
        [1, 1],
        [0, 1],
        [0.4, 0.4],
        [0.6, 0.6],
    ])

    dshape = img.shape[:2]
    use_unit_space = True
    warped, new_src = warp_image(img, c_src, c_dst, dshape=dshape, unit_space=use_unit_space)
    show_warped(img, warped, c_src, c_dst, unit_space=use_unit_space, new_src_pts=new_src)

    img = cv2.imread('1.jpg')
    c_src = np.array([[217, 39], [204, 95], [174, 223], [648, 402]]) # (x, y) in each row
    c_dst = np.array([[283, 54], [166, 101], [198, 250], [666, 372]])
    # c_src = c_src / np.array(img.shape[1::-1]).reshape(1, 2)
    # c_dst = c_dst / np.array(img.shape[1::-1]).reshape(1, 2)

    # dshape = (512, 512)
    dshape = img.shape[:2]
    use_unit_space = False

    lambd, uncertainty_sq = preprocess_uncertainty(c_src, c_dst)
    warped, new_src = warp_image(img, c_src, c_dst, dshape=dshape, unit_space=use_unit_space, lambd=lambd, uncertainty_square=uncertainty_sq)
    show_warped(img, warped, c_src, c_dst, unit_space=use_unit_space, new_src_pts=new_src)