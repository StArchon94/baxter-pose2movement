import numpy as np

def meshgrid(H, W):
    xs = np.linspace(0, W-1, W)
    ys = np.linspace(0, H-1, H)
    xs, ys = np.meshgrid(xs, ys)
    return xs, ys

class Camera:

    # def __init__(self, H, W, K, D, R, P):
    def __init__(self, H, W, K):
        self.H = H
        self.W = W
        self.K = np.array(K).reshape(3,3)
        # self.D = D
        # self.R = R
        # self.P = P

    def reconstruct(self, depth):
        H, W = depth.shape
        assert self.H == H
        assert self.W == W

        xs, ys = meshgrid(H, W)
        # print('xs shape')
        # print(xs.shape)
        ones = np.ones_like(xs)
        grid = np.stack([xs, ys, ones], axis=2)
        # print('grid shape')
        # print(grid.shape)
        depth = np.expand_dims(depth, axis=-1)
        grid = grid * depth
        grid = np.expand_dims(grid, axis=-1)
        # print('grid shape')
        # print(grid.shape)
        # print('np.linalg.inv(self.K)')
        # print(np.linalg.inv(self.K).shape)
        P = np.linalg.inv(self.K) @ grid
        P = P.squeeze()
        # print('P shape')
        # print(P.shape)

        return P