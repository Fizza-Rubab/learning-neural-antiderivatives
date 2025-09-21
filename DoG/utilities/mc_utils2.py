import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
# from utilities import build_2d_sampler_jax, build_3d_sampler_jax, build_1d_sampler_jax
import cv2
import trimesh
from pysdf import SDF
from skimage import measure
import plyfile
import logging
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator as RegularGridInterpolatorx




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


def gaussian(sigma):
    def kernel(x):
        return 1. / (sigma * jnp.sqrt(2*jnp.pi)) * jnp.exp(-(x) * (x) / (2 * sigma**2))
    return kernel


# @jax.jit
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

def pad_image(image, pad_fraction=0.25):
    height, width = image.shape[:2]
    pad_height = int(height * pad_fraction)
    pad_width = int(width * pad_fraction)
    padded_image = np.pad(image, ((0, 0), (pad_width, pad_width), (0,0)), mode='reflect')
    padded_image = np.pad(padded_image, ((pad_height, pad_height), (0, 0), (0,0)), mode='reflect')
    return padded_image



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

    function = kernel(half_size)
    vals = function(sample_points)
    values = jnp.prod(vals, -1, keepdims=True)

    return values, sample_points


def sample_kernel_gauss(half_size, index, shape, kernel):
    key = jrandom.PRNGKey(index)
    key, subkey = jrandom.split(key)
    sample_points = jrandom.uniform(key, shape) * (half_size + half_size) + (-half_size)

    function = kernel(half_size / 3)
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

        return carry + conv_out  # conv_out

    # ------------------------------------------------------------------------------------------------------------------

    convolution_results = jnp.zeros_like(data)
    return lax.fori_loop(0, sample_size, step, convolution_results) / sample_size


@click.command()
@click.option("--path", default='your_signal.npy', help="Path to 1D signal (.npy)")
@click.option("--sample_number", default=100, help="Number of MC samples")
@click.option("--save_path", default='../data', help="Path to save results")
@click.option("--half_size", default=0.3, help="Half-width of the kernel")
@click.option("--order", default=0, help="Order (for naming only)")
@click.option("--padding", default=0.25, help="Padding fraction of signal length")
@click.option("--kernel_type", default="minimal", help="gaussian or minimal")
def mc_conv_1d(path,
               sample_number,
               save_path,
               half_size,
               order,
               padding,
               kernel_type):
    signal = np.load(path)
    signal = signal.astype(np.float32)
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
    mc = mc_convolution(signal_padded,
                        sampling_grid,
                        sample_number,
                        half_size,
                        sampler,
                        sample_kernel,
                        (L, 1),
                        min1, 
                        1,
                        L)

    mc = np.array(mc)
    cropped_mc = mc[pad_len:N + pad_len, :]
    os.makedirs(save_path, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(path))[0]
    save_p = os.path.join(save_path, f'1d_order_{order}_{half_size}_samples_{sample_number}.npy')

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


@click.command()
@click.option("--path", default='/HPS/antiderivative_project/work/data/images/0278.png', help="path to save the results at")
@click.option("--sample_number", default=10000, help="sample number per pixels")
@click.option("--save_path", default='/HPS/n_ntumba/work/Fizza_project/temp_data/', help="path to save the results at")
@click.option("--half_size", default=0.03, help="Iterations per pixels")
@click.option("--order", default=0, help="Iterations per pixels")
@click.option("--kernel_type", default="minimal", help="gaussian or minimal")
@click.option("--padding", default=0.25, help="padding each side")
def mc_conv_2d(path,
               sample_number,
               save_path,
               half_size,
               order,
               kernel_type,
               padding):
    # ------------------------------------------------------------------------------------------------------------------

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
    mc = mc_convolution(image,
                        sampling_grid,
                        sample_number,
                        half_size,
                        sampler,
                        sample_kernel_gauss,
                        (image.shape[0], image.shape[1], 2),
                        gaussian,
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

    # from ios import send_to_tev
    # send_to_tev('mc', mc)


    save_p = os.path.join(save_path, f'image_order_{order}_{kernel_type}_{half_size}_samples_{sample_number}.npy')
    np.save(save_p, mc)

    # imageio.imsave(save_p[:-4] + '.png', (255*mc).astype(np.uint8))
    # imageio.imsave(save_p[:-4] + '_cropped.png', (255*cropped_mc).astype(np.uint8))




@click.command()
@click.option("--path",
              default='/HPS/n_ntumba/work/code relsease/code/neural-field-convolutions-by-repeated-differentiation/data/raw/geometry/armadillo.obj')
@click.option("--sample_number", default=5, help="sample number per pixels")
@click.option("--save_path", default='../data', help="path to save the results at")
@click.option("--half_size", default=0.01, help="Iterations per pixels")
@click.option("--order", default=0, help="Iterations per pixels")
def mc_conv_3d(path,
               sample_number,
               save_path,
               half_size,
               order):
    # ------------------------------------------------------------------------------------------------------------------

    voxel = mesh_to_sdf_tensor(path, 256)  # np.load(path, allow_pickle=True)
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

    mc = mc_convolution(voxel,
                        sampling_grid,
                        sample_number,
                        half_size,
                        sampler,
                        sample_kernel,
                        (voxel.shape[0], voxel.shape[1], voxel.shape[2], 3),
                        min0,
                        3,
                        voxel.shape[0])
    
    mc = np.array(mc)
    data = {
        'res': mc,
        'size': half_size,
        'samples': sample_number
    }

    save_p = os.path.join(save_path, f'3d_order_{order}_{half_size}_samples_{sample_number}.npy')
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


if __name__ == '__main__':
    # mc_conv_1d()
    mc_conv_2d()
    # mc_conv_3d() # for 3d mc generation
    # mc_conv_video() # for video mc generation
