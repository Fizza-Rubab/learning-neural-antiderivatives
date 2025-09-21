import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.spatial.transform import Rotation as R
import os



class GaussianMixture:
    def __init__(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.means = data['means']
        self.weights = data['weights']
        self.covs = data['covs']
        self.dim = self.means.shape[1] if self.means.ndim > 1 else 1
        self.threshold = data['threshold'] if 'threshold' in data else None
        self.discontinuous = bool(data['discontinuous']) if 'discontinuous' in data else False

    def eval(self, x):
        x = np.atleast_2d(x)
        result = np.zeros(x.shape[0])
        if self.dim == 1:
            for w, m, s in zip(self.weights, self.means, self.covs):
                result += w * norm.pdf(x.ravel(), m, s)
        else:
            for w, m, c in zip(self.weights, self.means, self.covs):
                result += w * multivariate_normal.pdf(x, m, c)
        if self.discontinuous and self.threshold is not None:
            if self.dim == 1:
                mask = (x.ravel() >= self.threshold)
            else:
                mask = np.all(x >= self.threshold, axis=1)
            result *= mask.astype(float)
        return result

class HyperrectangleMixture:
    def __init__(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.centers = data['centers']
        self.sizes = data['sizes']
        self.weights = data['weights']
        self.dim = self.centers.shape[1] if self.centers.ndim > 1 else 1
        self.rotation = bool(data['rotation']) if 'rotation' in data else False
        self.angles = data['angles'] if 'angles' in data else None
        self.rots = data['rotations'] if 'rotations' in data else None

    def eval(self, x):
        x = np.atleast_2d(x)
        result = np.zeros(x.shape[0])
        for i in range(len(self.weights)):
            center = self.centers[i]
            size = self.sizes[i]
            weight = self.weights[i]
            if self.dim == 1:
                inside = np.abs(x[:, 0] - center) <= size / 2
            elif self.dim == 2:
                rel = x - center
                if self.rotation and self.angles is not None:
                    angle = self.angles[i]
                    c, s = np.cos(-angle), np.sin(-angle)
                    rot = np.array([[c, -s], [s, c]])
                    rel = rel @ rot.T
                inside = np.all(np.abs(rel) <= size / 2, axis=1)
            elif self.dim == 3:
                rel = x - center
                if self.rotation and self.rots is not None:
                    rot = self.rots[i]
                    rel = rel @ rot.T
                inside = np.all(np.abs(rel) <= size / 2, axis=1)
            else:
                raise ValueError("Unsupported dimension")
            result += weight * inside.astype(float)
        return result
#
# # Plotting helpers
# def plot_1d(f, title, filename):
#     x = np.linspace(-1, 1, 1024).reshape(-1, 1)
#     y = f.eval(x)
#     plt.figure()
#     plt.plot(x, y)
#     plt.title(title)
#     plt.savefig(filename)
#     plt.close()
#
# def plot_2d(f, title, filename):
#     grid = np.linspace(-1, 1, 256)
#     xx, yy = np.meshgrid(grid, grid, indexing='ij')
#     coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)
#     zz = f.eval(coords).reshape(256, 256)
#     plt.figure()
#     plt.imshow(zz, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
#     plt.title(title)
#     plt.colorbar()
#     plt.savefig(filename)
#     plt.close()
#
# def plot_3d(f, title, filename):
#     grid = np.linspace(-1, 1, 64)
#     xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
#     coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
#     vals = f.eval(coords).reshape(64, 64, 64)
#     mid_slice = vals[:, :, 32]
#     plt.figure()
#     plt.imshow(mid_slice, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
#     plt.title(title + " (Z-slice)")
#     plt.colorbar()
#     plt.savefig(filename)
#     plt.close()
#
# base_path = "../../data/analytic_params"
# function_types = [
#     ("gm", GaussianMixture),
#     ("hr", HyperrectangleMixture)
# ]
#
# os.makedirs(base_path, exist_ok=True)
# for tag, FuncClass in function_types:
#     for dim in [1, 2, 3]:
#         file_path = os.path.join(base_path, f"{tag}_{dim}d_params.npz")
#         if os.path.exists(file_path):
#             f = FuncClass(file_path)
#             title = f"{tag.upper()} {dim}D"
#             filename = os.path.join(base_path, f"{tag}_{dim}d_eval_plot.png")
#
#             if dim == 1:
#                 plot_1d(f, title, filename)
#             elif dim == 2:
#                 plot_2d(f, title, filename)
#             elif dim == 3:
#                 plot_3d(f, title, filename)