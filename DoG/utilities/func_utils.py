import random
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sympy.diffgeom.rn import theta
import os
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from utilities import send_to_tev
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import random
from scipy.spatial.transform import Rotation as R

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

