import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import sys

sys.path.append('../')

import imageio
# from ismael.images.image_io import tev_display_image
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit
# import matplotlib.pyplot as plt
from jax import random as jrandom
from jax import lax
from jax import vmap
import time
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator
import sys
import click
import numpy as np
import os
import timeit
# from utilities import min0, min1, min2, gaussian
# from utilities import build_2d_sampler_jax, build_3d_sampler_jax, build_1d_sampler_jax, load_mp3
# from utilities import ackley_1d, ackley_2d, ackley_3d, gaussian_mixture_1d, gaussian_mixture_2d, gaussian_mixture_3d, mixture_hyperrectangles
# from ismael.images.image_io import send_to_tev
import cv2
import trimesh
from pysdf import SDF
from skimage import measure
import plyfile
import logging

import random
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sympy.diffgeom.rn import theta
import os
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from utilities import send_to_tev
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import random
from scipy.spatial.transform import Rotation as R
import glob
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator as RegularGridInterpolatorx
# from utilities import GaussianMixture, HyperrectangleMixture


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

def gaussian(sigma):
    def kernel(x):
        return 1. / (sigma * jnp.sqrt(2*jnp.pi)) * jnp.exp(-(x) * (x) / (2 * sigma**2))
    return kernel

import librosa
def load_mp3(filepath, sr=None, mono=True, normalize=True):
    signal, sr = librosa.load(filepath, sr=sr, mono=mono)
    if normalize:
        max_val = np.max(np.abs(signal)) + 1e-9
        signal = signal / max_val
    return signal.astype(np.float32)


def build_3d_sampler_jax(x_len, y_len, z_len, data):
    x = jnp.linspace(0, data.shape[0] - 1, x_len)
    y = jnp.linspace(0, data.shape[1] - 1, y_len)
    z = jnp.linspace(0, data.shape[2] - 1, z_len)
    return RegularGridInterpolatorx((x, y, z), data, bounds_error=False, fill_value=0.0)


def build_2d_sampler_jax(x_len, y_len, data):
    x = jnp.linspace(0, data.shape[0] - 1, x_len)
    y = jnp.linspace(0, data.shape[1] - 1, y_len)
    return RegularGridInterpolatorx((x, y), data, bounds_error=False, fill_value=0.0)


def build_1d_sampler_jax(x_len, shape, data):
    x = jnp.linspace(0, shape - 1, x_len)
    return RegularGridInterpolatorx((x,), data, bounds_error=False, fill_value=0.0)



def min0(s):
    return lambda x: jnp.where(
        jnp.abs(x) < s,
        jnp.ones_like(x) / (2 * s),
        jnp.zeros_like(x))

# @jax.jit
def min1(s):
    return lambda x: jnp.where(
        jnp.abs(x) < s,
        jnp.where(
            x < 0,
            (x + s) / s ** 2,
            (-x + s) / s ** 2,
        ),
        jnp.zeros_like(x))


def min2(s):
    basic_fct = lambda x: jnp.where(
        jnp.abs(x) <= 3,
        jnp.where(
            jnp.abs(x) <= 1,
            3 - x ** 2,
            jnp.where(
                x < -1,
                0.5 * (3 + x) ** 2,
                0.5 * (-3 + x) ** 2,
            ),
        ),
        jnp.zeros_like(x))
    return lambda x: basic_fct(x * 3 / s) * 3 / s / 8



def ackley_1d(x, a=20, b=0.2, c=4 * np.pi):
    """
    Computes the 1D Ackley function value at point x.

    The Ackley function is a widely used benchmark function for testing optimization algorithms.
    It features a nearly flat outer region and a large hole at the center, making it difficult to optimize.

    Parameters:
    ----------
    x : float or np.ndarray
        Input value(s) at which to evaluate the function.

    a : float, default=20
        Amplitude parameter affecting the global minimum depth.

    b : float, default=0.2
        Controls the decay rate of the exponential term.

    c : float, default=2π
        Frequency of the cosine term.

    Returns:
    -------
    float or np.ndarray
        Function value(s) at input x.
    """
    term1 = -a * np.exp(-b * np.sqrt(x ** 2))
    term2 = -np.exp(np.cos(c * x))
    return term1 + term2 + a + np.e


def ackley_2d(x, y, a=20, b=0.2, c=4 * np.pi):
    """
    Computes the 2D Ackley function value at point (x, y).

    Parameters:
    ----------
    x, y : float or np.ndarray
        Input values for each dimension.

    a : float, default=20
        Amplitude parameter affecting the global minimum depth.

    b : float, default=0.2
        Controls the decay rate of the exponential term.

    c : float, default=2π
        Frequency of the cosine term.

    Returns:
    -------
    float or np.ndarray
        Function value(s) at input (x, y).
    """
    term1 = -a * np.exp(-b * np.sqrt((x ** 2 + y ** 2) / 2))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    return term1 + term2 + a + np.e


def ackley_3d(x, y, z, a=20, b=0.2, c=4 * np.pi):
    """
    Computes the 3D Ackley function value at point (x, y, z).

    Parameters:
    ----------
    x, y, z : float or np.ndarray
        Input values for each dimension.

    a : float, default=20
        Amplitude parameter affecting the global minimum depth.

    b : float, default=0.2
        Controls the decay rate of the exponential term.

    c : float, default=2π
        Frequency of the cosine term.

    Returns:
    -------
    float or np.ndarray
        Function value(s) at input (x, y, z).
    """
    term1 = -a * np.exp(-b * np.sqrt((x ** 2 + y ** 2 + z ** 2) / 3))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y) + np.cos(c * z)) / 3)
    return term1 + term2 + a + np.e


def gaussian_mixture_1d(num_components=3,
                        seed=None,
                        random_weights=False,
                        stds_range=(0.05, 0.333),
                        weight_range=(0, 1),
                        range_discontinuity=(-1, 1),
                        discontinuous=False):
    """
    Creates a 1D Gaussian Mixture Model (GMM) probability density function (PDF) with the specified parameters.

    Parameters:
    ----------
    num_components : int, default=3
        The number of Gaussian components in the mixture.

    seed : int or None, default=None
        Random seed for reproducibility. If None, randomness is not seeded.

    random_weights : bool, default=False
        If True, assigns random weights to the mixture components drawn from `weight_range`.
        If False, uses equal weights (i.e., all ones).

    stds_range : tuple of float, default=(0.05, 0.333)
        The range (min, max) from which standard deviations of the Gaussian components are sampled uniformly.

    weight_range : tuple of float, default=(0, 1)
        The range (min, max) from which weights are sampled if `random_weights` is True.

    Returns:
    -------
    pdf : function
        A function `pdf(x)` that computes the probability density at one or more points `x`.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    means = np.random.uniform(-1, 1, num_components)
    variances = np.random.uniform(stds_range[0], stds_range[1], num_components)
    weights = np.random.uniform(weight_range[0], weight_range[1], num_components) if random_weights else np.ones(
        num_components)

    def pdf(x):
        x = np.atleast_1d(x)
        result = np.zeros_like(x)
        for w, m, s in zip(weights, means, variances):
            result += w * norm.pdf(x, m, s)

        if discontinuous:
            threshold = np.random.uniform(range_discontinuity[0], range_discontinuity[1])
            mask = np.where(x < threshold, 0, 1)
            result *= mask

        return result

    return pdf


def sample_2d_covariance_matrix(dim, var_range=(1, 3)):
    """
    Samples a random 2D covariance matrix with variances and correlation in specified ranges.

    This function generates a symmetric, positive-definite 2×2 covariance matrix based on:
    - Two variances sampled uniformly from `var_range`.
    - A correlation coefficient sampled uniformly from [-0.9, 0.9].

    Parameters:
    ----------
    dim : int
        Dimensionality of the covariance matrix. Must be 2.
        (Included for compatibility; only 2D matrices are supported.)

    var_range : tuple of float, default=(1, 3)
        Range (min, max) for sampling the variances along the diagonal.

    Returns:
    -------
    cov : np.ndarray of shape (2, 2)
        A 2D positive-definite covariance matrix.
    """
    var1, var2 = np.random.uniform(var_range[0], var_range[1], size=dim)  # variance sampling
    rho = np.random.uniform(-0.9, 0.9)  # correlation sampling

    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)
    cov = np.array([
        [var1,                rho * std1 * std2],
        [rho * std1 * std2,   var2]
    ])

    return cov



def gaussian_mixture_2d(num_components=3,
                        seed=None,
                        random_weights=False,
                        variances_range=(0.05, 0.5),
                        weight_range=(0, 1),
                        range_discontinuity=(-1, 1),
                        discontinuous=False):
    """
    Creates a 2D Gaussian Mixture Model (GMM) probability density function (PDF).

    This function generates a mixture of 2D Gaussian components, each with a randomly sampled
    mean and covariance matrix. Weights can be fixed or randomly sampled.

    Parameters:
    ----------
    num_components : int, default=3
        Number of Gaussian components in the mixture.

    seed : int or None, default=None
        Random seed for reproducibility. If None, randomness is not seeded.

    random_weights : bool, default=False
        If True, component weights are sampled uniformly from `weight_range`.
        If False, all components are assigned equal weight.

    variances_range : tuple of float, default=(0.05, 0.5)
        Range (min, max) from which the variances of the covariance matrices are sampled.
        Used internally by `sample_2d_covariance_matrix`.

    weight_range : tuple of float, default=(0, 1)
        Range (min, max) used to sample the weights if `random_weights` is True.

    Returns:
    -------
    pdf : function
        A function `pdf(xy)` that evaluates the GMM at one or more 2D input points.
        - `xy` should be an array-like of shape (n_samples, 2).
        - Returns a NumPy array of shape (n_samples,) with the density values.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    means = np.random.uniform(-1, 1, size=(num_components, 2))
    covs = [sample_2d_covariance_matrix(2, (variances_range[0], variances_range[1])) for _ in range(num_components)]
    weights = np.random.uniform(weight_range[0], weight_range[1], num_components) if random_weights else np.ones(num_components)


    def pdf(xy):
        xy = np.atleast_2d(xy)
        result = np.zeros(xy.shape[0])
        for w, m, c in zip(weights, means, covs):
            result += w * multivariate_normal.pdf(xy, m, c)

        if discontinuous:
            threshold = np.random.uniform(range_discontinuity[0], range_discontinuity[1], (2))
            mask = np.where((xy[:, 0] < threshold[0]) | (xy[:, 1] < threshold[1]), 0, 1)
            result = result * mask

        return result

    return pdf


def sample_3d_covariance_matrix(dim=2, var_range=(1, 3)):
    """
    Samples a random positive-definite covariance matrix for multivariate Gaussian distributions.

    This function generates a symmetric, positive-definite matrix by multiplying a random matrix
    with its transpose, and then scaling it so that the maximum eigenvalue lies within the specified
    `var_range`. The result is a valid covariance matrix suitable for Gaussian sampling.

    Parameters:
    ----------
    dim : int, default=2
        Dimensionality of the covariance matrix. For 3D applications, set `dim=3`.

    var_range : tuple of float, default=(1, 3)
        Desired range for the largest eigenvalue (i.e., the overall variance scale) of the covariance matrix.
        The matrix is scaled such that its largest eigenvalue is sampled uniformly within this range.

    Returns:
    -------
    Sigma : np.ndarray of shape (dim, dim)
        A symmetric, positive-definite covariance matrix.
    """
    A = np.random.randn(dim, dim)  # random matrix with standard normal entries
    Sigma = A @ A.T  # ensures symmetric positive-definite matrix
    max_eig = np.linalg.eigvalsh(Sigma)[-1]  # largest eigenvalue
    scale = np.random.uniform(*var_range)
    scale = scale / max_eig  # normalize to match desired variance scale
    return Sigma * scale



def gaussian_mixture_3d(num_components=3,
                        seed=None,
                        random_weights=False,
                        variances_range=(0.4, 1.5),
                        weight_range=(0, 1),
                        range_discontinuity=(-1, 1),
                        discontinuous=False):
    """
    Creates a 3D Gaussian Mixture Model (GMM) probability density function (PDF).

    This function generates a mixture of 3D Gaussian distributions with randomly sampled means
    and covariance matrices. Weights for the components can either be fixed or randomly sampled.

    Parameters:
    ----------
    num_components : int, default=3
        Number of Gaussian components in the mixture.

    seed : int or None, default=None
        Random seed for reproducibility. If None, randomness is not seeded.

    random_weights : bool, default=False
        If True, component weights are sampled uniformly from `weight_range`.
        If False, all components are assigned equal weights.

    variances_range : tuple of float, default=(0.4, 1.5)
        Range (min, max) from which the variances of the 3D covariance matrices are sampled.

    weight_range : tuple of float, default=(0, 1)
        Range (min, max) used to sample the weights if `random_weights` is True.

    Returns:
    -------
    pdf : function
        A function `pdf(xyz)` that evaluates the GMM at one or more 3D input points.
        - `xyz` should be an array-like of shape (n_samples, 3).
        - Returns a NumPy array of shape (n_samples,) containing the density values.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    means = np.random.uniform(-1, 1, size=(num_components, 3))
    covs = [sample_3d_covariance_matrix(3, (variances_range[0], variances_range[1])) for _ in range(num_components)]
    weights = np.random.uniform(weight_range[0], weight_range[1], num_components) if random_weights else np.ones(num_components)

    def pdf(xyz):
        xyz = np.atleast_2d(xyz)
        result = np.zeros(xyz.shape[0])
        for w, m, c in zip(weights, means, covs):
            result += w * multivariate_normal.pdf(xyz, m, c)

        if discontinuous:
            threshold = np.random.uniform(range_discontinuity[0], range_discontinuity[1], (3))
            mask = np.where((xyz[:, 0] < threshold[0]) |
                            (xyz[:, 1] < threshold[1]) |
                            (xyz[:, 2] < threshold[2]), 0, 1)
            result = mask * result

        return result

    return pdf


def point_in_hyperrectangle2d(x, center, size, value_inside=1, value_outside=0, angle=0.0, rotation=False):
    """
    Determines whether 2D point(s) lie within a rotated or axis-aligned rectangle.

    This function checks whether each input point lies within a 2D hyperrectangle
    (i.e., rectangle), optionally applying rotation. The rectangle is defined by its
    center, size (width and height), and rotation angle.

    Parameters:
    ----------
    x : array-like of shape (n_samples, 2) or (2,)
        The 2D point(s) to check. Can be a single point or a batch of points.

    center : array-like of shape (2,)
        The center coordinates (x, y) of the rectangle.

    size : array-like of shape (2,)
        The side lengths (width, height) of the rectangle along its local axes.

    value_inside : float, default=1
        Value to assign to points that fall inside the rectangle.

    value_outside : float, default=0
        Value to assign to points that fall outside the rectangle.

    angle : float, default=0.0
        Rotation angle of the rectangle in radians. Used only if `rotation=True`.

    rotation : bool, default=False
        Whether to apply rotation to the rectangle. If False, the rectangle is axis-aligned.

    Returns:
    -------
    result : np.ndarray
        An array of shape (n_samples,) indicating whether each point is inside the rectangle.
        Values are either `value_inside` or `value_outside`.

    Notes:
    -----
    - The rotation is applied **clockwise** to align points with the local axes of the rotated rectangle.
    - Input `x` is automatically reshaped to 2D if needed. If a single point is passed, the output will still be a NumPy array of length 1.
    """
    center = np.array(center)
    size = np.array(size)
    rel = x - center

    if rotation:
        c, s = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        rel = rel @ rot.T

    inside = np.all(np.abs(rel) <= size / 2, axis=1)
    result = np.where(inside, value_inside, value_outside)

    return result


def point_in_hyperrectangle3d(x, center, size, value_inside=1, value_outside=0, rot=None, rotation=False):
    """
    Determines whether 3D point(s) lie within a rotated or axis-aligned cuboid (hyperrectangle).

    This function checks if each input point falls within a specified 3D rectangular region
    (cuboid), optionally applying a rotation matrix to the points relative to the cuboid's orientation.

    Parameters:
    ----------
    x : array-like of shape (n_samples, 3)
        The 3D point(s) to evaluate. Should be a NumPy array of points in 3D space.

    center : array-like of shape (3,)
        The center coordinates (x, y, z) of the cuboid.

    size : array-like of shape (3,)
        The side lengths (width, height, depth) of the cuboid along its local axes.

    value_inside : float, default=1
        Value to return for each point that lies inside the cuboid.

    value_outside : float, default=0
        Value to return for each point that lies outside the cuboid.

    rot : np.ndarray of shape (3, 3), optional
        A 3D rotation matrix that defines the cuboid's orientation. Only used if `rotation=True`.

    rotation : bool, default=False
        Whether to apply the rotation matrix to align points with the rotated cuboid.

    Returns:
    -------
    result : np.ndarray of shape (n_samples,)
        An array where each entry is either `value_inside` or `value_outside`,
        depending on whether the corresponding input point is inside the cuboid.

    Notes:
    -----
    - This function expects `x` to be a NumPy array of shape `(n_samples, 3)`.
    - The rotation (if enabled) transforms the relative coordinates into the cuboid's local frame.
    - No check is made to verify that `rot` is a valid rotation matrix; the caller must ensure this.
    """
    center = np.array(center)
    size = np.array(size)
    rel = x - center

    if rotation:
        rel = rel @ rot.T

    inside = np.all(np.abs(rel) <= size / 2, axis=1)
    result = np.where(inside, value_inside, value_outside)
    return result


def point_in_hyperrectangle1d(x, center, size, value_inside=1, value_outside=0):
    """
    Determines whether 1D point(s) lie within an interval (1D hyperrectangle).

    Parameters:
    ----------
    x : array-like
        Input 1D point(s) to evaluate.

    center : float or array-like
        Center of the interval.

    size : float or array-like
        Length of the interval.

    value_inside : float, default=1
        Value returned for points inside the interval.

    value_outside : float, default=0
        Value returned for points outside the interval.

    Returns:
    -------
    result : np.ndarray
        Array indicating for each point whether it is inside (`value_inside`) or outside (`value_outside`) the interval.

    Note:
    -----
    - The function expects `x` to be broadcast-compatible with `center` and `size`.
    - The original code uses `np.all` with `axis=1`, so `x` should be at least 2D.
    """
    center = np.array(center)
    size = np.array(size)
    rel = x - center

    inside = np.all(np.abs(rel) <= size / 2, axis=1)
    result = np.where(inside, value_inside, value_outside)
    return result



def mixture_hyperrectangles(xs, dim,
                            num_rects=3,
                            seed=None,
                            random_weights=False,
                            sizes_range=(0.1, 0.5),
                            angle_range=(0, 2 * np.pi),
                            rotation=False):
    """
    Evaluates a mixture of axis-aligned or rotated hyperrectangles in 1D, 2D, or 3D.

    This function generates multiple hyperrectangles with random centers, sizes, and (optionally)
    rotations. It evaluates whether points `xs` fall inside any of these hyperrectangles,
    returning a weighted sum indicating the degree of membership.

    Parameters:
    ----------
    xs : np.ndarray
        Input points to evaluate. Shape should be (n_samples, dim) or (dim,) for a single point.

    dim : int
        Dimensionality of the space (1, 2, or 3). Controls the geometry and function used.

    num_rects : int, default=3
        Number of hyperrectangles in the mixture.

    seed : int or None, default=None
        Random seed for reproducibility. If None, randomness is not seeded.

    random_weights : bool, default=False
        If True, assigns random weights to each hyperrectangle. Otherwise, all weights are 1.

    sizes_range : tuple of float, default=(0.1, 0.5)
        Range (min, max) of side lengths for each dimension of the hyperrectangles.

    angle_range : tuple of float, default=(0, 2 * np.pi)
        Range of rotation angles (in radians) for 2D rectangles. Only used if `dim == 2`.

    rotation : bool, default=False
        If True, applies random rotations to 2D and 3D hyperrectangles.
        If False, the rectangles are axis-aligned.

    Returns:
    -------
    final : float or np.ndarray
        A scalar or array of shape (n_samples,) representing the weighted sum of
        hyperrectangle membership values. Each value is between 0 and the sum of weights.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    centers = np.random.uniform(-1, 1, (num_rects, dim))
    sizes = np.random.uniform(sizes_range[0], sizes_range[1], (num_rects, dim))
    angles = np.random.uniform(angle_range[0], angle_range[1], size=num_rects)
    rotations = [R.random().as_matrix() for _ in range(num_rects)]
    weights = np.random.uniform(0, 1, num_rects) if random_weights else np.ones(num_rects)

    final = 0
    for i in range(num_rects):
        if dim == 2:
            final += weights[i] * point_in_hyperrectangle2d(xs, centers[i], sizes[i], angle=angles[i], rotation=rotation)
        elif dim == 3:
            final += weights[i] * point_in_hyperrectangle3d(xs, centers[i], sizes[i], rot=rotations[i], rotation=rotation)
        elif dim == 1:
            final += weights[i] * point_in_hyperrectangle1d(xs, centers[i], sizes[i])

    return final


def pad_image(image, pad_fraction=0.3):
    height, width = image.shape[:2]
    pad_height = int(height * pad_fraction)
    pad_width = int(width * pad_fraction)
    padded_image = np.pad(image, ((0, 0), (pad_width, pad_width), (0,0)), mode='reflect')
    padded_image = np.pad(padded_image, ((pad_height, pad_height), (0, 0), (0,0)), mode='reflect')
    return padded_image


def pad_signal(signal, pad_fraction=0.3):
    length = signal.shape[0]
    pad_length = int(length * pad_fraction)
    padded_signal = np.pad(signal.flatten(), (pad_length, pad_length), mode='reflect')
    return padded_signal[:, None]

def pad_sdf(sdf_volume, pad_fraction=0.3, constant_value=1.0):
    depth, height, width = sdf_volume.shape[:3]
    pad_depth = int(depth * pad_fraction)
    pad_height = int(height * pad_fraction)
    pad_width = int(width * pad_fraction)
    padding = ((pad_depth, pad_depth), (pad_height, pad_height), (pad_width, pad_width))
    if len(sdf_volume.shape) > 3:
        padding = padding + ((0, 0),) * (len(sdf_volume.shape) - 3)
    padded_sdf = np.pad(sdf_volume, padding, mode='constant', constant_values=constant_value)
    return padded_sdf



def sdf_to_ply_and_save(
        sdf_tensor,
        vozel_origin,
        voxel_size,
        output_file,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply

    :param sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = np.array(sdf_tensor)  # .numpy()
    print("dims: ", numpy_3d_sdf_tensor.ndim)

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = measure.marching_cubes(numpy_3d_sdf_tensor, level=0, spacing=[voxel_size] * 3)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.show()
    except Exception as e:
        # pass
        print("exception thrown", e)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = vozel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = vozel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = vozel_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    print('vert', num_verts, 'face', num_faces)

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append((faces[i, :].tolist(),))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (output_file))
    ply_data.write(output_file)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def mesh_to_sdf_tensor(mesh_path, resolution):
    def scale_to_unit_cube(mesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

        vertices = mesh.vertices - mesh.bounding_box.centroid
        vertices *= 2 / np.max(mesh.bounding_box.extents)
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

    mesh = trimesh.load_mesh(mesh_path)

    # convert mesh to sdf
    mesh = scale_to_unit_cube(mesh)
    sdf = SDF(mesh.vertices, mesh.faces)

    x = np.linspace(-1, 1, resolution)
    grid = np.stack(np.meshgrid(x, x, x), -1)
    sampling_grid = np.reshape(grid, (-1, 3))
    output = -sdf(sampling_grid)

    level_set = 0
    sdf_tensor = output.reshape(resolution, resolution, resolution)
    return sdf_tensor


def save_mesh(voxel, save_path, file_name='mesh.ply'):
    voxel_origin = [-1] * 3
    voxel_size = 2.0 / (voxel.shape[0] - 1)

    if len(voxel.shape) > 3:
        voxel = voxel[..., 0]

    vertices, faces, normals, _ = measure.marching_cubes(voxel, level=0, spacing=[voxel_size] * 3)

    sdf_to_ply_and_save(
        voxel,
        voxel_origin,
        voxel_size,
        os.path.join(save_path, file_name),
        None,
        None,
    )


def load_frames(video_path, res, resize=False):
    frame_names = os.listdir(video_path)
    frame_names.sort()
    frames = []

    print(f'Loading ...')
    for i in range(len(frame_names)):
        current_path = os.path.join(video_path, frame_names[i])
        current_frame = imageio.imread(current_path) / 255.0
        frame = current_frame

        if resize:
            frame = cv2.resize(frame, res)[..., None, :]
            frames.append(frame)
        else:
            frames.append(frame[..., None, :])

    return np.concatenate(frames, axis=2)


def sample_kernel(half_size, index, shape, kernel):
    key = jrandom.PRNGKey(index)
    key, subkey = jrandom.split(key)
    sample_points = jrandom.uniform(key, shape) * (half_size + half_size) + (-half_size)

    function = kernel(half_size / 3) # the division is only valid for gaussians.
    vals = function(sample_points)
    values = jnp.prod(vals, -1, keepdims=True)

    return values, sample_points


def mc_convolution(data,
                   sampling_grid,
                   sample_size,
                   half_size,
                   signal_sampler,
                   kernel_sampler,
                   shape,
                   kernel,
                   dimension,
                   max_dim):

    def step(index, carry):
        # ------------------------------------------------------------------------------------------------------------------

        kernel_values, sample_points = kernel_sampler(half_size, index, shape, kernel)
        current_sample_points = (sample_points * max_dim) / 2

        if len(data.shape) == 4 and data.shape[3] == 3:
            coord_x = sampling_grid[..., :1]
            coord_y = sampling_grid[..., 1:2]
            coord_z = sampling_grid[..., 2:] + current_sample_points
            coord_z = jnp.clip(coord_z, 0, max_dim - 1)
            shifted_coordinates = jnp.concatenate([coord_x, coord_y, coord_z], -1)
        else:
            shifted_coordinates = sampling_grid + current_sample_points
            shifted_coordinates = jnp.clip(shifted_coordinates, 0, max_dim - 1)

        sampled_signal = signal_sampler(shifted_coordinates)
        conv_out = (sampled_signal * kernel_values) * ((half_size - (-half_size)) ** dimension)
        print("Sampled signal shape:", sampled_signal.shape)
        print("Kernel values shape:", kernel_values.shape)
        print("Convolution output shape:", conv_out.shape)


        return carry + conv_out  # conv_out

    # ------------------------------------------------------------------------------------------------------------------

    convolution_results = jnp.zeros_like(data)
    return lax.fori_loop(0, sample_size, step, convolution_results) / sample_size


@click.command()
@click.option("--path", default='your_signal.wav', help="Path to 1D signal (.wav)")
@click.option("--sample_number", default=100, help="Number of MC samples")
@click.option("--save_path", default='../data', help="Path to save results")
@click.option("--half_size", default=0.3, help="Half-width of the kernel")
@click.option("--order", default=0, help="Order (for naming only)")
@click.option("--padding", default=0.3, help="Padding fraction of signal length")
@click.option("--kernel_type", default="minimal", help="gaussian or minimal")
def mc_conv_1d(path,
               sample_number,
               save_path,
               half_size,
               order,
               padding,
               kernel_type):
    # signal = np.load(path)
    signal = load_mp3(path)
    signal = signal.astype(np.float32)
    signal = pad_signal(signal, padding)
    if signal.ndim > 1:
        signal = signal.squeeze()
    N = signal.shape[0]
    pad_len = int(padding * N)
    signal_padded = np.pad(signal, (pad_len, pad_len), mode='reflect')
    signal_padded = jnp.array(signal_padded)[..., None]  # shape (L, 1)
    L = signal_padded.shape[0]
    x = jnp.linspace(0, L - 1, L)
    sampling_grid = x[:, None]
    sampler = build_1d_sampler_jax(signal_padded.shape[0], signal_padded.shape[0], signal_padded)

    if kernel_type=="minimal":
        if order==0:
            kernel = min0
        elif order==1:
            kernel = min1
        else:
            kernel = min2
    else:
        kernel = gaussian


    mc = mc_convolution(signal_padded,
                        sampling_grid,
                        sample_number,
                        half_size,
                        sampler,
                        sample_kernel,
                        (L, 1),
                        kernel, 
                        1,
                        L)

    mc = np.array(mc)
    cropped_mc = mc[pad_len:N + pad_len, :]
    os.makedirs(save_path, exist_ok=True)


    out_name = os.path.splitext(os.path.basename(path))[0]
    base = os.path.splitext(os.path.basename(path))[0]
    save_p = os.path.join(save_path, f'{base}_1d_order_{order}_{half_size}_samples_{sample_number}.npy')

    np.save(save_p, {
        'res': mc,
        'size': half_size,
        'samples': sample_number,
        'padding': padding
    })

    print(f"Saved: {save_p}")
    plt.plot(cropped_mc, label='MC Convolved')
    plt.title(f'MC 1D Convolution: {out_name}')
    plt.tight_layout()
    plt.show()


def normalize_array(x, out_min=0, out_max=1):
    in_min, in_max = np.min(x), np.max(x)
    return (out_max - out_min) / (in_max - in_min) * (x - in_min) + out_min

def read_pose(file_path, normalize=True, num_frames=5000):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith("Skeletool")]

    pose_list = []
    for line in data_lines:
        parts = line.split()
        coords = list(map(float, parts[1:]))
        frame_pose = np.array(coords).reshape(-1, 3)
        pose_list.append(frame_pose)

    pose_array = np.stack(pose_list)  
    total_frames = pose_array.shape[0]
    if num_frames is not None and num_frames < total_frames:
        start = (total_frames - num_frames) // 2
        pose_array = pose_array[start:start + num_frames]
    elif num_frames is not None and num_frames > total_frames:
        raise ValueError(f"Requested {num_frames} frames, but only {total_frames} available.")

    if normalize:
        pose_array = normalize_array(pose_array)
    print(pose_array.shape)
    return pose_array


@click.command()
@click.option("--path", default='../../data/humanposes/1', help="Path to folder of pose .csv files")
@click.option("--sample_number", default=100000, help="Number of MC samples")
@click.option("--save_path", default='../../data/motion', help="Path to save results")
@click.option("--half_size", default=0.04, help="Half-width of the kernel")
@click.option("--order", default=0, help="Order (for naming only)")
@click.option("--padding", default=0.3, help="Padding fraction of sequence length")
@click.option("--kernel_type", default="minimal", help="gaussian or minimal")
def mc_conv_motion_1d(path,
                      sample_number,
                      save_path,
                      half_size,
                      order,
                      padding,
                      kernel_type):
    st = time.time()
    # Load and normalize motion data: shape (T, J, C)
    signal = read_pose(path).astype(np.float32).reshape(-1, 69)

    # Padding disabled (can be re-enabled):
    # from utilities import pad_signal
    # signal = pad_signal(signal, padding)

    # Convert to JAX array: shape (T, J, C)
    signal_jax = jnp.array(signal)
    N, C = signal.shape  # N = time, C = 69 (flattened joints)
    x = jnp.linspace(0, N - 1, N)
    sampling_grid = x[:, None]  # (N, 1)

    sampler = build_1d_sampler_jax(N, N, signal_jax)

    # Select kernel
    if kernel_type == "minimal":
        if order == 0:
            kernel = min0
        elif order == 1:
            kernel = min1
        elif order == 2:
            kernel = min2
        else:
            raise ValueError("Unsupported order for minimal kernel.")

    elif kernel_type == "gaussian":
        kernel = gaussian
    else:
        raise ValueError("Unknown kernel type.")

    mc = mc_convolution(signal_jax,
                        sampling_grid,
                        sample_number,
                        half_size,
                        sampler,
                        sample_kernel,
                        (N, 1),
                        kernel,
                        1,
                        N)
    mc.block_until_ready()
    mc = np.array(mc)  # shape (N, 69)
    et = time.time()
    print("Elapsed time", et - st)
    os.makedirs(save_path, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(path))[0]
    save_path_full = os.path.join(save_path,
        f'{out_name}_motion1d_order_{order}_{kernel_type}_{half_size}_samples_{sample_number}.npy'
    )

    np.save(save_path_full, {
        'res': mc,
        'size': half_size,
        'samples': sample_number,
        'padding': padding,
        "time": et - st
    })

    print(f"Saved convolved motion to: {save_path_full}")


def select_kernel(kernel_type, order):
    if kernel_type == "minimal":
        return [min0, min1, min2][min(order, 2)]
    else:
        return gaussian

def save_numpy(result, path, base, dim, order, size, samples, padding):

    file_name = f"{base}_{dim}_order{order}_{size}_samples{samples}.npy"
    np.save(os.path.join(path, file_name), {
        'res': result,
        'size': size,
        'samples': samples,
        'padding': padding
    })
    print("Saved:", file_name)


@click.command()
@click.option("--analytic", default='ackley', help="ackley, gm, or hr")
@click.option("--sample_number", default=100)
@click.option("--save_path", default='../../data/analytic/1d')
@click.option("--half_size", default=0.3)
@click.option("--order", default=0)
@click.option("--grid_size", default=2048)
@click.option("--padding", default=0.3)
@click.option("--kernel_type", default="minimal")
@click.option("--seed", default=100)
def mc_conv_1d_analytic(analytic, sample_number, save_path, half_size, order, grid_size, padding, kernel_type, seed):

    os.makedirs(save_path, exist_ok=True)
    print(f'analytics : {analytic}')
    np.random.seed(seed)
    if analytic == "ackley":
        f = ackley_1d
    elif analytic == "gm":
        path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/gm_1d_params.npz'
        f = GaussianMixture(path).eval

    elif analytic == "hr":
        path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/hr_1d_params.npz'
        f = HyperrectangleMixture(path).eval

    else:
        raise ValueError("Unsupported function")

    x = jnp.linspace(-1, 1, grid_size).reshape(-1, 1)
    signal = f(x).reshape(-1, 1)
    signal = np.array(signal)

    # signal = pad_signal(signal, padding)
    L = signal.shape[0]
    signal = jnp.array(signal)
    sampling_grid = jnp.linspace(0, L - 1, L)[:, None]

    sampler = build_1d_sampler_jax(L, L, signal)
    kernel = select_kernel(kernel_type, order)

    mc = mc_convolution(signal, sampling_grid, sample_number, half_size,
                        sampler, sample_kernel, (L, 1), kernel, 1, L)
    result = np.array(mc)
    save_numpy(result, save_path, analytic, "1d", order, half_size, sample_number, padding)


@click.command()
@click.option("--analytic", default='ackley')
@click.option("--sample_number", default=100)
@click.option("--save_path", default='../../data/analytic/2d')
@click.option("--half_size", default=0.3)
@click.option("--order", default=0)
@click.option("--grid_size", default=1024)
@click.option("--padding", default=0.3)
@click.option("--kernel_type", default="minimal")
@click.option("--seed", default=100)
def mc_conv_2d_analytic(analytic, sample_number, save_path, half_size, order, grid_size, padding, kernel_type, seed):
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')

    if analytic == "ackley":
        image = ackley_2d(xx, yy)[..., None]  # shape (H, W, 1)

    elif analytic == "gm":
        path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/gm_2d_params.npz'
        f = GaussianMixture(path).eval
        coords = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)
        image = f(coords).reshape(grid_size, grid_size, -1)

    elif analytic == "hr":
        path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/hr_2d_params.npz'
        f = HyperrectangleMixture(path).eval
        coords = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)
        image = f(coords).reshape(grid_size, grid_size, -1)

    else:
        raise ValueError("Unsupported function")

    image = np.array(image)
    # image = pad_image(image, padding)
    H, W, _ = image.shape

    image = jnp.array(image)
    sampling_grid = jnp.stack(jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'), -1)

    sampler = build_2d_sampler_jax(H, W, image) 
    kernel = select_kernel(kernel_type, order)

    mc = mc_convolution(image, sampling_grid, sample_number, half_size,
                        sampler, sample_kernel, (H, W, 2), kernel, 2, H)
    result = np.array(mc)
    save_numpy(result, save_path, analytic, "2d", order, half_size, sample_number, padding)


@click.command()
@click.option("--analytic", default='ackley')
@click.option("--sample_number", default=100)
@click.option("--save_path", default='../../data/analytic/3d')
@click.option("--half_size", default=0.05)
@click.option("--order", default=0)
@click.option("--grid_size", default=512)
@click.option("--padding", default=0.3)
@click.option("--kernel_type", default="minimal")
@click.option("--seed", default=100)
def mc_conv_3d_analytic(analytic, sample_number, save_path, half_size, order, grid_size, padding, kernel_type, seed):
    st = time.time()
    coords = [jnp.linspace(-1, 1, grid_size) for _ in range(3)]
    xx, yy, zz = jnp.meshgrid(*coords, indexing='ij')

    if analytic == "ackley":
        volume = ackley_3d(xx, yy, zz)[..., None]  # shape (grid, grid, grid, 1)

    elif analytic == "gm":
        path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/gm_3d_params.npz'
        f = GaussianMixture(path).eval

        pts = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        volume = f(pts).reshape(grid_size, grid_size, grid_size, -1)

    elif analytic == "hr":
        path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/hr_3d_params.npz'
        f = HyperrectangleMixture(path).eval

        pts = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        volume = f(pts).reshape(grid_size, grid_size, grid_size, -1)

    else:
        raise ValueError("Unsupported function")


    volume = np.array(volume)
    # volume = pad_sdf(volume[..., 0], padding)[..., None]

    H, W, D, _ = volume.shape
    volume = jnp.array(volume)
    sampling_grid = jnp.stack(jnp.meshgrid(jnp.arange(H), jnp.arange(W), jnp.arange(D), indexing='ij'), -1)

    sampler = build_3d_sampler_jax(H, W, D, volume)
    kernel = select_kernel(kernel_type, order)

    mc = mc_convolution(volume, sampling_grid, sample_number, half_size,
                        sampler, sample_kernel, (H, W, D, 3), kernel, 3, H)

    et = time.time()
    print("Elapsed time", et - st)
    result = np.array(mc)

    os.makedirs(save_path, exist_ok=True)

    z = result.shape[2] // 2
    slice2d = result[:, :, z, 0]
    plt.figure()
    plt.imshow(slice2d, cmap='viridis', origin='lower', aspect='auto')
    plt.title(f"{analytic} 3D MC - Z Slice {z}")
    plt.colorbar()
    image_path = os.path.join(save_path, f"{analytic}_3d_mc_slice_z{z}.png")
    plt.savefig(image_path)
    plt.close()
    print("Saved slice image:", image_path)
    
    save_numpy(result, save_path, analytic, "3d", order, half_size, sample_number, padding)



def pad_envmap(envmap, pad_fraction=0.3):
    H, W, C = envmap.shape
    pad_h = int(H * pad_fraction)
    pad_w = int(W * pad_fraction)
    left_pad = envmap[:, -pad_w:, :]
    right_pad = envmap[:, :pad_w, :]
    padded_w = np.concatenate([left_pad, envmap, right_pad], axis=1)
    top_pad = np.flip(padded_w[:pad_h, :, :], axis=[0, 1])
    bottom_pad = np.flip(padded_w[-pad_h:, :, :], axis=[0, 1])
    padded_envmap = np.concatenate([top_pad, padded_w, bottom_pad], axis=0)
    return padded_envmap



@click.command()
@click.option("--path", default='/HPS/n_ntumba/work/network_fitting/Progressive experiments/Production experiments/gts/images/256/1.jpg', help="path to save the results at")
@click.option("--sample_number", default=100, help="sample number per pixels")
@click.option("--save_path", default='../data', help="path to save the results at")
@click.option("--half_size", default=0.3, help="Iterations per pixels")
@click.option("--order", default=0, help="Iterations per pixels")
@click.option("--padding", default=0.3, help="padding each side")
@click.option("--kernel_type", default="minimal", help="gaussian or minimal")
def mc_conv_env(path,
               sample_number,
               save_path,
               half_size,
               order,
               kernel_type,
               padding):
    # ------------------------------------------------------------------------------------------------------------------
    # save_path+=f"_mc_order={order}"
    st = time.time()
    image =  np.array(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB))
    Ho, Wo, D = image.shape
    image = pad_envmap(image, padding)
    image = jnp.array(image)
    H, W, D = image.shape

    # create sampling coordinates
    x = jnp.linspace(0, H - 1, H)
    y = jnp.linspace(0, W - 1, W)
    grid = jnp.meshgrid(x, y, indexing='ij')
    sampling_grid = jnp.stack(grid, -1)

    sampler = build_2d_sampler_jax(image.shape[0], image.shape[1], image)
    if kernel_type=="minimal":
        if order==0:
            kernel = min0
        elif order==1:
            kernel = min1
        else:
            kernel = min2
    else:
        kernel = gaussian
    print("kernel", kernel)
    mc = mc_convolution(image,
                        sampling_grid,
                        sample_number,
                        half_size,
                        sampler,
                        sample_kernel,
                        (image.shape[0], image.shape[1], 2),
                        kernel,
                        2,
                        image.shape[0])
    mc.block_until_ready()
    et = time.time()

    mc = np.array(mc)
    cropped_mc = mc[int(padding*Ho): Ho + int(padding*Ho), int(padding*Wo): Wo + int(padding*Wo), :]
    data = {
        'res': mc,
        'size': half_size,
        'samples': sample_number,
        "time": et - st,
        "padding": padding
    }
    print("Time taken:", data["time"])
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(save_path, exist_ok=True)
    save_p = os.path.join(save_path, f'{base}_2d_order_{order}_{kernel_type}_{half_size}_samples_{sample_number}.npy')
    np.save(save_p, data)
    # imageio.imsave(save_p[:-4] + '.png', (255*mc).astype(np.uint8))
    # imageio.imsave(save_p[:-4] + '_cropped.png', (255*cropped_mc).astype(np.uint8))



@click.command()
@click.option("--path", default='/HPS/n_ntumba/work/network_fitting/Progressive experiments/Production experiments/gts/images/256/1.jpg', help="path to save the results at")
@click.option("--sample_number", default=100, help="sample number per pixels")
@click.option("--save_path", default='../data', help="path to save the results at")
@click.option("--half_size", default=0.3, help="Iterations per pixels")
@click.option("--order", default=0, help="Iterations per pixels")
@click.option("--padding", default=0.3, help="padding each side")
@click.option("--kernel_type", default="minimal", help="gaussian or minimal")
def mc_conv_2d(path,
               sample_number,
               save_path,
               half_size,
               order,
               kernel_type,
               padding):
    # ------------------------------------------------------------------------------------------------------------------
    # save_path+=f"_mc_order={order}"
    st = time.time()
    image = imageio.v3.imread(path) / 255
    Ho, Wo, D = image.shape
    image = pad_image(image, padding)
    image = jnp.array(image)
    H, W, D = image.shape

    # create sampling coordinates
    x = jnp.linspace(0, H - 1, H)
    grid = jnp.meshgrid(x, x, indexing='ij')
    sampling_grid = jnp.stack(grid, -1)

    sampler = build_2d_sampler_jax(image.shape[0], image.shape[1], image)
    if kernel_type=="minimal":
        if order==0:
            kernel = min0
        elif order==1:
            kernel = min1
        else:
            kernel = min2
    else:
        kernel = gaussian
    mc = mc_convolution(image,
                        sampling_grid,
                        sample_number,
                        half_size,
                        sampler,
                        sample_kernel,
                        (image.shape[0], image.shape[1], 2),
                        kernel,
                        2,
                        image.shape[0])
    mc.block_until_ready()
    et = time.time()

    mc = np.array(mc)
    cropped_mc = mc[int(padding*Ho): Ho + int(padding*Ho), int(padding*Wo): Wo + int(padding*Wo), :]
    data = {
        'res': mc,
        'size': half_size,
        'samples': sample_number,
        "time": et - st,
        "padding": padding
    }
    print("Time taken:", data["time"])
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(save_path, exist_ok=True)
    save_p = os.path.join(save_path, f'{base}_2d_order_{order}_{kernel_type}_{half_size}_samples_{sample_number}.npy')
    np.save(save_p, data)
    # imageio.imsave(save_p[:-4] + '.png', (255*mc).astype(np.uint8))
    # imageio.imsave(save_p[:-4] + '_cropped.png', (255*cropped_mc).astype(np.uint8))




@click.command()
@click.option("--path",
              default='/HPS/n_ntumba/work/code relsease/code/neural-field-convolutions-by-repeated-differentiation/data/raw/geometry/armadillo.obj')
@click.option("--sample_number", default=5, help="sample number per pixels")
@click.option("--save_path", default='../data', help="path to save the results at")
@click.option("--half_size", default=0.01, help="Iterations per pixels")
@click.option("--order", default=0, help="Iterations per pixels")
@click.option("--padding", default=0.3, help="padding each side")
@click.option("--kernel_type", default="minimal", help="gaussian or minimal")
def mc_conv_3d(path,
               sample_number,
               save_path,
               half_size,
               order,
               kernel_type,
               padding):
    # ------------------------------------------------------------------------------------------------------------------
    # save_path+=f"_mc_order={order}"
    st = time.time()
    voxel = mesh_to_sdf_tensor(path, 256)  # np.load(path, allow_pickle=True)
    voxel = pad_sdf(voxel, padding)
    voxel = jnp.array(voxel)[..., None]
    H, W, D, _ = voxel.shape

    # create sampling coordinates
    x = jnp.linspace(0, H - 1, H)
    grid = jnp.meshgrid(x, x, x, indexing='ij')
    sampling_grid = jnp.stack(grid, -1)

    sampler = build_3d_sampler_jax(voxel.shape[0],
                                   voxel.shape[1],
                                   voxel.shape[2],
                                   voxel)
    if kernel_type=="minimal":
        if order==0:
            kernel = min0
        elif order==1:
            kernel = min1
        else:
            kernel = min2
    else:
        kernel = gaussian
    mc = mc_convolution(voxel,
                        sampling_grid,
                        sample_number,
                        half_size,
                        sampler,
                        sample_kernel,
                        (voxel.shape[0], voxel.shape[1], voxel.shape[2], 3),
                        kernel,
                        3,
                        voxel.shape[0])
    mc.block_until_ready()
    mc = np.array(mc)
    et = time.time()
    print("Elapsed time", et - st)
    data = {
        'res': mc,
        'size': half_size,
        'samples': sample_number,
        "time": et - st
    }

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(save_path, exist_ok=True)
    save_p = os.path.join(save_path, f'{base}_3d_order_{order}_{half_size}_samples_{sample_number}.npy')
    np.save(save_p, data)



# todo finish video mc
# todo train all models once to make sure all is working fine
@click.command()
@click.option("--path", default='/HPS/n_ntumba/work/image_data/video/newest/coals/NFC/dice_temp/')
@click.option("--resolution", default=32, help="sample number per pixels")
@click.option("--sample_number", default=100, help="sample number per pixels")
@click.option("--save_path", default='../data', help="path to save the results at")
@click.option("--half_size", default=0.4, help="Iterations per pixels")
@click.option("--order", default=0, help="Iterations per pixels")
def mc_conv_video(path,
                  resolution,
                  sample_number,
                  save_path,
                  half_size,
                  order):
    # ------------------------------------------------------------------------------------------------------------------
    video = load_frames(path, (None, None), resize=False)
    video = np.float32(video)
    video = jnp.array(video)
    H, W, D, C = video.shape

    # create sampling coordinates
    x = jnp.linspace(0, H - 1, H)
    d = jnp.linspace(0, D - 1, D)

    grid = jnp.meshgrid(x, x, d, indexing='ij')
    sampling_grid = jnp.stack(grid, -1)
    sampler = build_3d_sampler_jax(video.shape[0],
                                   video.shape[1],
                                   video.shape[2],
                                   video)

    mc = mc_convolution(video,
                        sampling_grid,
                        sample_number,
                        half_size,
                        sampler,
                        sample_kernel,
                        (1, 1, sampling_grid.shape[2], 1),
                        min0,
                        1,
                        video.shape[2])

    mc = np.array(mc)
    mc = jnp.reshape(mc, video.shape)

    data = {
        'res': mc,
        'size': half_size,
        'samples': sample_number
    }

    save_p = os.path.join(save_path, f'video_order_{order}_{half_size}_samples_{sample_number}.npy')
    np.save(save_p, data)


@click.group()
def cli():
    pass

cli.add_command(mc_conv_1d, name='audio')
cli.add_command(mc_conv_2d, name='image')
cli.add_command(mc_conv_env, name='env')
cli.add_command(mc_conv_3d, name='geometry')
cli.add_command(mc_conv_video, name='video')
cli.add_command(mc_conv_1d_analytic, name='analytic1d')
cli.add_command(mc_conv_2d_analytic, name='analytic2d')
cli.add_command(mc_conv_3d_analytic, name='analytic3d')
cli.add_command(mc_conv_motion_1d, name='motion')


if __name__ == '__main__':
    cli()
