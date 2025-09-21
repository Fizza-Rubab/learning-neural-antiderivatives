import os.path
import imageio
import torch
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator as RegularGridInterpolatorx
import shutil
from functools import reduce
from .minimal_kernels import minimal_kernel_diracs
from ._kernel import TempKernel2d, Kernel3d, TempKernel1d
import simpleimageio as sio
import librosa
import numpy as np


def create_or_recreate_folders(folder):
    """
    deletes existing folder if they already exist and
    recreates then. Only valid for training mode. does not work in
    resume mode
    :return:
    """
    # if os.path.isdir(folder):
    # shutil.rmtree(folder)
    # os.mkdir(folder)
    if not os.path.isdir(folder):
        os.makedirs(folder)


def load_mp3(filepath, sr=None, mono=True, normalize=True):
    signal, sr = librosa.load(filepath, sr=sr, mono=mono)
    if normalize:
        max_val = np.max(np.abs(signal)) + 1e-9
        signal = signal / max_val
    return signal.astype(np.float32)


def load_montecarlo_gt(path):
    monte_carlo_ground_truth = np.load(path, allow_pickle=True)
    monte_carlo_ground_truth = monte_carlo_ground_truth.item()['res']
    return monte_carlo_ground_truth


def create_minimal_filter_2d(order, half_size=1.0):
    diracs_x, diracs_y = minimal_kernel_diracs(order, half_size)
    grid = np.stack(np.meshgrid(diracs_x, diracs_x), -1)
    values = np.outer(diracs_y, diracs_y)

    kernel = TempKernel2d()  # Kernel2d()
    kernel.initialize_control_points(grid, values, order)
    return kernel


def create_minimal_kernel_3d(args):
    kernelxs, kernel_ys = minimal_kernel_diracs(0, 1 / args.kernel_scale)
    values = reduce(np.multiply.outer, (kernel_ys, kernel_ys, kernel_ys))
    coords = np.stack(np.meshgrid(kernelxs, kernelxs, kernelxs), -1)

    kernel = Kernel3d()
    kernel.initialize_control_points(coords, values)

    return kernel


def create_minimal_filter_1d(order, half_size=1.0):
    diracs_x, diracs_y = minimal_kernel_diracs(order, half_size)
    kernel = TempKernel1d()
    kernel.initialize_control_points(diracs_x, diracs_y, order)
    return kernel


def map_range(values, old_range, new_range):
    NewRange = (new_range[0] - new_range[1])
    OldRange = (old_range[0] - old_range[1])
    new_values = (((values - old_range[0]) * NewRange) / OldRange) + new_range[0]
    return new_values


def build_1d_sampler(x_len, shape, data, method='linear'):
    x = np.linspace(0, shape - 1, x_len)
    return RegularGridInterpolatorx((x,), data, method=method)


def build_2d_sampler(x_len, y_len, data, method='linear'):
    x = np.linspace(0, data.shape[0] - 1, x_len)
    y = np.linspace(0, data.shape[1] - 1, y_len)
    return RegularGridInterpolator((y, x), data, method=method)


def build_3d_sampler(x_len, y_len, z_len, data):
    x = np.linspace(-1, 1, x_len)
    y = np.linspace(-1, 1, y_len)
    t = np.linspace(-1, 1, z_len)
    return RegularGridInterpolator((x, y, t), data)


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


def calculate_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    psnr = -10.0 * torch.log10(mse + 1e-8)
    return psnr.item()


def tensor_to_image(tensor, *, denormalize=True):
    """
    Converts a torch tensor image to a numpy array image.

    Parameters
    ----------
    tensor : torch.Tensor
        The image as a torch tensor of shape (channels, height, width) or (1, channels, height, width).
    denormalize : bool
        If true, transform the data range from [-1, 1] to [0, 1].

    Returns
    -------
    img : np.ndarray
        The image as a numpy array of shape (height, width, channels).
    """

    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError("If the image tensor has a batch dimension, it must have length 1.")
        tensor = tensor[0]
    if denormalize:
        tensor = (tensor + 1) * 0.5
    img = tensor.numpy(force=True)
    return np.transpose(img, (1, 2, 0))


# one function to handle all tev displays
def send_to_tev(name, tensor, gamma=1.0):
    check_tev_connection()

    # Supporting functions

    # prepare image for display in tev (gamma correction applied)
    def clean_image(img):
        if img.flags['C_CONTIGUOUS']:
            # This condition is necessary to correct how the image is ordered in memory which in turn corrects the "stride" parameter for use inside tev
            img = np.asfortranarray(img)

        if img.ndim == 2:  # if grayscale
            img = img[:, :, None]

        if img.ndim > 2 and img.shape[2] == 2:  # in case of 2 channels, append a third one
            img = np.dstack((img, np.zeros(img.shape[:-1])))

        img = np.power(img, gamma)
        return np.ascontiguousarray(img)

    #==============================================

    # prepare layered image for display in tev
    def clean_layered_image(layers):
        assert type(layers) is dict, "layers needs to be a dictionary of the form \"layer_name\" : layer"
        layers_clean = {}
        for id, img in layers.items():
            layers_clean[id] = clean_image(img)
        return layers_clean

    #==============================================

    # send an image to tev
    def tev_display_image(name, img):
        with sio.TevIpc() as tev:
            tev.display_image(name, clean_image(img))

    #==============================================

    # send a layered image to tev
    def tev_display_layered_image(name, layers):
        with sio.TevIpc() as tev:
            tev.display_layered_image(name, clean_layered_image(layers))

    #==============================================

    # display a (batched, multi-channel, or both) image tensor (torch / numpy) as a layered image in tev
    def tev_display_tensor(name, tensor):
        # helper function for tev display strings
        def create_str(idx, channels):
            if idx == channels:
                string = f"{idx:02}"
            elif channels - idx == 1:
                string = f"{idx:02},{idx + 1:02}"
            elif channels - idx == 2:
                string = f"{idx:02},{idx + 1:02},{idx + 2:02}"
            else:
                string = f"{idx:02},{idx + 1:02},{idx + 2:02},{idx + 3:02}"

            return string

        # builds the dictionary of RGBA images given a single/multi-channel image/tensor
        def construct_layers(image, batch_name=""):
            layers = {}
            channels = image.shape[2]
            for idx in range(0, channels, 4):
                layers[f"{batch_name}[{create_str(idx, channels - 1)}]"] = image[:, :, idx:idx + 4]

            return layers

        # process a torch tensor and a ndarray differently before construct_layers
        def process_tensor(tensor):
            return tensor_to_image(tensor.unsqueeze(0)) if torch.is_tensor(tensor) else tensor

        layers = {}
        for idx, layer in enumerate(tensor):
            l = construct_layers(process_tensor(layer), f"B_{idx:02}_")
            layers.update(l)

        tev_display_layered_image(name, layers)

    #==============================================

    if isinstance(tensor, dict):
        try:
            tev_display_layered_image(name, tensor)
        except:
            raise ValueError("Expected a dictionary with key: string; value: ndarray.")

    elif isinstance(tensor, np.ndarray):
        assert tensor.ndim <= 4, "Need numpy array in format (batch (optional), channels, width, height)"
        if tensor.ndim <= 3:
            # numpy array has the format (width, height, channels)"
            tev_display_image(name, tensor)
        else:
            # numpy array has the format (batch, width, height, channels)"
            tev_display_tensor(name, tensor)

    elif isinstance(tensor, torch.Tensor):
        assert tensor.dim() == 4, "Need tensor in format (batch, channels, width, height)"
        tev_display_tensor(name, tensor)

    else:
        raise ValueError("Input tensor must be a dictionary, ndarray, or Torch Tensor.")


#==============================================

# Check if tev is initialized
def check_tev_connection():
    import socket
    tev = sio.TevIpc()
    tev._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Attempt to create a socket and connect to the specified host and port
    with tev._socket as s:
        try:
            s.connect((tev._hostname, tev._port))
        except:
            raise ConnectionError("Please Initialize Tev !")

        #==============================================


# close an image in tev
def tev_close_image(name):
    with sio.TevIpc() as tev:
        tev.close_image(name)
