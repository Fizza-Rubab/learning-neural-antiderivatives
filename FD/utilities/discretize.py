import numpy as np
import jax.numpy as jnp
import click
import os

from func_utils import (
    ackley_1d, ackley_2d, ackley_3d,
    gaussian_mixture_1d, gaussian_mixture_2d, gaussian_mixture_3d,
    mixture_hyperrectangles
)

def save_numpy(result, path, base, dim, order, size, samples, padding):
    os.makedirs(path, exist_ok=True)
    file_name = f"{base}_{dim}.npy"
    np.save(os.path.join(path, file_name), result)
    print("Saved:", file_name)
    

import matplotlib.pyplot as plt

def save_signal_image(signal, save_path, analytic, dimension):
    os.makedirs(save_path, exist_ok=True)

    if dimension == 1:
        plt.figure()
        plt.plot(signal[:, 0])
        plt.title(f"{analytic} 1D")
        plt.xlabel("Grid Index")
        plt.ylabel("Value")
        image_path = os.path.join(save_path, f"{analytic}_1d.png")

    elif dimension == 2:
        plt.figure()
        plt.imshow(signal[:, :, 0], cmap='viridis', origin='lower', aspect='auto')
        plt.title(f"{analytic} 2D")
        plt.colorbar()
        image_path = os.path.join(save_path, f"{analytic}_2d.png")

    elif dimension == 3:
        # Create coordinate grid like your example
        x = np.linspace(-1, 1, signal.shape[0])
        y = np.linspace(-1, 1, signal.shape[1])
        z = np.linspace(-1, 1, signal.shape[2])
        coordinates_3d = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1)

        mid_z = signal.shape[2] // 2
        coords_2d = coordinates_3d[:, :, mid_z, :]
        Z3d = signal[:, :, mid_z, 0]
        fig = plt.figure()
        plt.imshow(Z3d, cmap='viridis', origin='lower')
        plt.title(f"{analytic} 3D - Z Slice (imshow)")
        plt.colorbar()
        image_path = os.path.join(save_path, f"{analytic}_3d.png")


    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    print("Saved image:", image_path)


@click.command()
@click.option("--analytic", default='ackley', help="ackley, gm, or hr")
@click.option("--grid_size", default=256)
@click.option("--save_path", default="../../data/analytic_discrete")
@click.option("--seed", default=100)
@click.option("--dimension", default=1, type=int, help="1, 2, or 3")
def discretize_analytic_signal(analytic, grid_size, save_path, seed, dimension):
    np.random.seed(seed)

    if dimension == 1:
        x = jnp.linspace(-1, 1, grid_size).reshape(-1, 1)

        if analytic == "ackley":
            signal = ackley_1d(x)
        elif analytic == "gm":
            f = gaussian_mixture_1d(seed=seed)
            signal = f(x)
        elif analytic == "hr":
            f = lambda x: mixture_hyperrectangles(x, dim=1, seed=seed)
            signal = f(x)
        else:
            raise ValueError("Unsupported function")

        signal = np.array(signal).reshape(-1, 1)

    elif dimension == 2:
        x = jnp.linspace(-1, 1, grid_size)
        y = jnp.linspace(-1, 1, grid_size)
        xx, yy = jnp.meshgrid(x, y, indexing='ij')
        coords = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)

        if analytic == "ackley":
            signal = ackley_2d(xx, yy)
        elif analytic == "gm":
            f = gaussian_mixture_2d(seed=seed)
            signal = f(coords).reshape(grid_size, grid_size)
        elif analytic == "hr":
            f = lambda x: mixture_hyperrectangles(x, dim=2, seed=seed, num_rects=5, rotation=True)
            signal = f(coords).reshape(grid_size, grid_size)
        else:
            raise ValueError("Unsupported function")

        signal = np.array(signal)[..., None]  

    elif dimension == 3:
        coords = [jnp.linspace(-1, 1, grid_size) for _ in range(3)]
        xx, yy, zz = jnp.meshgrid(*coords, indexing='ij')
        pts = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        if analytic == "ackley":
            signal = ackley_3d(xx, yy, zz)
        elif analytic == "gm":
            f = gaussian_mixture_3d(seed=seed)
            signal = f(pts).reshape(grid_size, grid_size, grid_size)
        elif analytic == "hr":
            f = lambda x: mixture_hyperrectangles(x, dim=3, seed=seed, num_rects=45)
            signal = f(pts).reshape(grid_size, grid_size, grid_size)
        else:
            raise ValueError("Unsupported function")

        signal = np.array(signal)[..., None]  
    else:
        raise ValueError("Unsupported dimension")

    # Save
    os.makedirs(save_path, exist_ok=True)
    save_numpy(signal, save_path, analytic, dimension, None, None, None, None)
    save_signal_image(signal, save_path, analytic, dimension)



if __name__ == "__main__":
    discretize_analytic_signal()



# import numpy as np
# import os
# from scipy.spatial.transform import Rotation as R
# import random

# def sample_2d_covariance_matrix(var_range=(1, 3)):
#     """Sample 2D covariance matrix - matches the original function exactly"""
#     var1, var2 = np.random.uniform(var_range[0], var_range[1], size=2)
#     rho = np.random.uniform(-0.9, 0.9)
#     std1 = np.sqrt(var1)
#     std2 = np.sqrt(var2)
#     return np.array([[var1, rho * std1 * std2],
#                      [rho * std1 * std2, var2]])

# def sample_3d_covariance_matrix(var_range=(1, 3)):
#     """Sample 3D covariance matrix - matches the original function exactly"""
#     A = np.random.randn(3, 3)
#     Sigma = A @ A.T
#     max_eig = np.linalg.eigvalsh(Sigma)[-1]
#     scale = np.random.uniform(*var_range) / max_eig
#     return Sigma * scale

# def save_gaussian_mixture_params(dim, seed, num_components, out_path, 
#                                  var_range, weight_range=(0, 1), random_weights=False, 
#                                  range_discontinuity=(-1, 1), discontinuous=False):
#     """
#     Save Gaussian Mixture parameters using the EXACT same random sequence as the original functions.
#     This ensures perfect alignment between saved parameters and discretized functions.
#     """
#     # CRITICAL: Match the exact seeding sequence from the original functions
#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)
    
#     # Generate parameters using the same sequence as the original functions
#     means = np.random.uniform(-1, 1, size=(num_components, dim))
    
#     if dim == 1:
#         # For 1D: original function samples variances directly
#         variances = np.random.uniform(var_range[0], var_range[1], num_components)
#         covs = variances  # Store as 1D array for 1D case
#     elif dim == 2:
#         # For 2D: sample covariance matrices
#         covs = np.array([sample_2d_covariance_matrix(var_range) for _ in range(num_components)])
#     elif dim == 3:
#         # For 3D: sample covariance matrices  
#         covs = np.array([sample_3d_covariance_matrix(var_range) for _ in range(num_components)])
#     else:
#         raise ValueError("Unsupported dimension")
    
#     weights = np.random.uniform(weight_range[0], weight_range[1], num_components) if random_weights else np.ones(num_components)
    
#     # Handle discontinuous case
#     threshold = None
#     if discontinuous:
#         if dim == 1:
#             threshold = np.random.uniform(range_discontinuity[0], range_discontinuity[1])
#         else:
#             threshold = np.random.uniform(range_discontinuity[0], range_discontinuity[1], dim)
    
#     # Save parameters
#     os.makedirs(out_path, exist_ok=True)
#     filename = os.path.join(out_path, f"gm_{dim}d_params.npz")
    
#     save_dict = {
#         'means': means,
#         'covs': covs, 
#         'weights': weights,
#         'num_components': num_components,
#         'discontinuous': discontinuous,
#         'range_discontinuity': range_discontinuity
#     }
    
#     if threshold is not None:
#         save_dict['threshold'] = threshold
        
#     np.savez(filename, **save_dict)
#     print(f"Saved GM {dim}D parameters to {filename}")

# def save_hyperrectangle_params(dim, seed, num_rects, out_path,
#                                size_range=(0.1, 0.5), angle_range=(0, 2*np.pi),
#                                random_weights=False, rotation=False):
#     """
#     Save hyperrectangle parameters using the same random sequence as the original function.
#     """
#     # CRITICAL: Match the exact seeding sequence
#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)
    
#     # Generate parameters in the same order as the original function
#     centers = np.random.uniform(-1, 1, (num_rects, dim))
#     sizes = np.random.uniform(size_range[0], size_range[1], (num_rects, dim))
#     angles = np.random.uniform(angle_range[0], angle_range[1], size=num_rects)
#     rotations = [R.random().as_matrix() for _ in range(num_rects)]
#     weights = np.random.uniform(0, 1, num_rects) if random_weights else np.ones(num_rects)
    
#     # Prepare save dictionary based on dimension
#     save_dict = {
#         'centers': centers,
#         'sizes': sizes, 
#         'weights': weights,
#         'num_rects': num_rects,
#         'rotation': rotation
#     }
    
#     if dim == 2:
#         save_dict['angles'] = angles
#     elif dim == 3:
#         save_dict['rotations'] = np.array(rotations)
    
#     # Save parameters
#     os.makedirs(out_path, exist_ok=True)
#     filename = os.path.join(out_path, f"hr_{dim}d_params.npz")
#     np.savez(filename, **save_dict)
#     print(f"Saved HR {dim}D parameters to {filename}")

# # Usage example with the exact same parameters as your discretization
# def save_all_params():
#     """Save parameters for all function types and dimensions"""
#     base_path = "../../data/analytic_params"  # Match your discretization path structure
#     seed = 100  # Same seed as discretization
    
#     # 1D functions
#     print("Saving 1D parameters...")
#     save_gaussian_mixture_params(
#         dim=1, seed=seed, num_components=3, out_path=base_path, 
#         var_range=(0.05, 0.333)  # Match discretization stds_range
#     )
#     save_hyperrectangle_params(
#         dim=1, seed=seed, num_rects=3, out_path=base_path
#     )
    
#     # 2D functions  
#     print("Saving 2D parameters...")
#     save_gaussian_mixture_params(
#         dim=2, seed=seed, num_components=3, out_path=base_path,
#         var_range=(0.05, 0.5)  # Match discretization variances_range
#     )
#     save_hyperrectangle_params(
#         dim=2, seed=seed, num_rects=5, out_path=base_path,
#         rotation=True  # Match discretization
#     )
    
#     # 3D functions
#     print("Saving 3D parameters...")
#     save_gaussian_mixture_params(
#         dim=3, seed=seed, num_components=3, out_path=base_path,
#         var_range=(0.4, 1.5)  # Match discretization variances_range
#     )
#     save_hyperrectangle_params(
#         dim=3, seed=seed, num_rects=45, out_path=base_path,
#         rotation=False  # Match discretization (no rotation specified = False)
#     )

# if __name__ == "__main__":
#     save_all_params()

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm, multivariate_normal
# from scipy.spatial.transform import Rotation as R


# def eval_gm_1d(x, means, covs, weights):
#     y = np.zeros_like(x)
#     for m, s, w in zip(means, covs, weights):
#         y += w * norm.pdf(x, m, s)
#     return y

# def eval_gm_2d(xx, yy, means, covs, weights):
#     coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)
#     z = np.zeros(len(coords))
#     for m, c, w in zip(means, covs, weights):
#         z += w * multivariate_normal.pdf(coords, m, c)
#     return z.reshape(xx.shape)

# def eval_gm_3d(xx, yy, zz, means, covs, weights):
#     coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
#     z = np.zeros(len(coords))
#     for m, c, w in zip(means, covs, weights):
#         z += w * multivariate_normal.pdf(coords, m, c)
#     return z.reshape(xx.shape)


# def point_in_hr(x, center, size, angle=None, rot=None, rotation=False):
#     rel = x - center
#     if rotation:
#         if rot is not None:
#             rel = rel @ rot.T
#         elif angle is not None:
#             c, s = np.cos(-angle), np.sin(-angle)
#             R2 = np.array([[c, -s], [s, c]])
#             rel = rel @ R2.T
#     return np.all(np.abs(rel) <= size / 2, axis=1)


# def eval_hr(x, centers, sizes, weights, dim, rotation=False, angles=None, rots=None):
#     out = np.zeros(x.shape[0])
#     for i in range(len(centers)):
#         if dim == 1:
#             inside = np.abs(x[:, 0] - centers[i, 0]) <= sizes[i, 0] / 2
#         elif dim == 2:
#             inside = point_in_hr(x, centers[i], sizes[i], angle=angles[i] if angles is not None else 0.0, rotation=rotation)
#         elif dim == 3:
#             inside = point_in_hr(x, centers[i], sizes[i], rot=rots[i] if rots is not None else None, rotation=rotation)
#         out += weights[i] * inside
#     return out


# def plot_and_save_1d(x, y, out_path, name):
#     plt.figure()
#     plt.plot(x, y)
#     plt.title(name)
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_path, f"{name}.png"))
#     plt.close()

# def plot_and_save_2d(xx, yy, zz, out_path, name):
#     plt.figure()
#     plt.imshow(zz, origin='lower', extent=[-1, 1, -1, 1], cmap='viridis')
#     plt.title(name)
#     plt.colorbar()
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_path, f"{name}.png"))
#     plt.close()


# def main():
#     base_path = "../../data/analytic_params"
#     dims = [1, 2, 3]

#     for dim in dims:
#         # ---------- GM ----------
#         gm_path = os.path.join(base_path, f"gm_{dim}d_params.npz")
#         if os.path.exists(gm_path):
#             data = np.load(gm_path)
#             means = data['means']
#             covs = data['covs']
#             weights = data['weights']
#             name = f"gm_{dim}d_plot"

#             if dim == 1:
#                 x = np.linspace(-1, 1, 1024)
#                 y = eval_gm_1d(x, means, covs, weights)
#                 plot_and_save_1d(x, y, base_path, name)
#             elif dim == 2:
#                 grid = np.linspace(-1, 1, 256)
#                 xx, yy = np.meshgrid(grid, grid, indexing='ij')
#                 zz = eval_gm_2d(xx, yy, means, covs, weights)
#                 plot_and_save_2d(xx, yy, zz, base_path, name)
#             elif dim == 3:
#                 grid = np.linspace(-1, 1, 64)
#                 xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
#                 z = eval_gm_3d(xx, yy, zz, means, covs, weights)
#                 mid_z = z.shape[2] // 2
#                 plot_and_save_2d(xx[:, :, 0], yy[:, :, 0], z[:, :, mid_z], base_path, name)

#         # ---------- HR ----------
#         hr_path = os.path.join(base_path, f"hr_{dim}d_params.npz")
#         if os.path.exists(hr_path):
#             data = np.load(hr_path)
#             centers = data['centers']
#             sizes = data['sizes']
#             weights = data['weights']
#             rotation = data['rotation'].item()
#             angles = data['angles'] if 'angles' in data else None
#             rots = data['rotations'] if 'rotations' in data else None
#             name = f"hr_{dim}d_plot"

#             if dim == 1:
#                 x = np.linspace(-1, 1, 1024).reshape(-1, 1)
#                 y = eval_hr(x, centers, sizes, weights, dim)
#                 plot_and_save_1d(x[:, 0], y, base_path, name)
#             elif dim == 2:
#                 grid = np.linspace(-1, 1, 256)
#                 xx, yy = np.meshgrid(grid, grid, indexing='ij')
#                 coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)
#                 z = eval_hr(coords, centers, sizes, weights, dim, rotation, angles=angles)
#                 plot_and_save_2d(xx, yy, z.reshape(256, 256), base_path, name)
#             elif dim == 3:
#                 grid = np.linspace(-1, 1, 64)
#                 xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
#                 coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
#                 z = eval_hr(coords, centers, sizes, weights, dim, rotation, rots=rots)
#                 plot_and_save_2d(xx[:, :, 0], yy[:, :, 0], z.reshape(64, 64, 64)[:, :, 32], base_path, name)


# if __name__ == "__main__":
#     main()
