import numpy as np


def meshgrid(H, W):
    xs = np.linspace(0, W - 1, W)
    ys = np.linspace(0, H - 1, H)
    xs, ys = np.meshgrid(xs, ys)
    return xs, ys


class Camera:
    def __init__(self, H, W, K):
        xs, ys = meshgrid(H, W)
        ones = np.ones_like(xs)
        self.grid = np.stack([xs, ys, ones], axis=2)
        K = np.array(K).reshape(3, 3)
        self.K_inv = np.linalg.inv(K)

    def reconstruct(self, depth):
        depth = np.expand_dims(depth, axis=-1)
        grid = self.grid * depth
        grid = np.expand_dims(grid, axis=-1)
        P = self.K_inv @ grid
        P = P.squeeze()
        return P
