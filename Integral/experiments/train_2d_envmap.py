import os
import sys
sys.path.append('../')
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from model import CoordinateNet_ordinary as CoordinateNet
from utilities import TrainingLog
import jax
import jax.numpy as jnp
from jax import random, jit, lax
from functools import partial
from PIL import Image
import cv2
from scipy.stats import qmc
from utilities import ackley_2d_jnp, gaussian_mixture_2d_jnp, mixture_hyperrectangles_jnp, GaussianMixture, HyperrectangleMixture
import time
import math

def jax_interp2d(xy_query, image):
    H, W, C = image.shape
    x = jnp.clip((xy_query[:, 0] + 1.0) * 0.5 * (W - 1), 0, W - 2)
    y = jnp.clip((xy_query[:, 1] + 1.0) * 0.5 * (H - 1), 0, H - 2)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0
    wy = y - y0

    def get_pixel(ix, iy):
        return image[iy, ix]

    Ia = jax.vmap(get_pixel)(x0, y0)
    Ib = jax.vmap(get_pixel)(x1, y0)
    Ic = jax.vmap(get_pixel)(x0, y1)
    Id = jax.vmap(get_pixel)(x1, y1)

    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy

    return wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id


def monte_carlo_antiderivative_2d_flattened(x, y, a, c, order, num_samples):
    factorial = math.factorial(order - 1)

    # Sample Sobol points in [0, 1]^2
    sampler = qmc.Sobol(d=2, scramble=True)
    sobol = sampler.random_base2(m=int(np.log2(num_samples)))
    sobol_samples = jnp.array(sobol)  # (num_samples, 2)

    # Rescale to [a,x] × [c,y]
    t_x = a + sobol_samples[:, 0] * (x - a)
    t_y = c + sobol_samples[:, 1] * (y - c)

    # Evaluate f(s, t) at these locations
    f_vals = jax.vmap(lambda sx, sy: f(sx, sy))(t_x, t_y)  # (num_samples, D)
    kernel_x = (x - t_x) ** (order - 1)
    kernel_y = (y - t_y) ** (order - 1)
    kernel = (kernel_x * kernel_y)[:, None] / (factorial ** 2)  # (num_samples, 1)

    return (x - a) * (y - c) * jnp.mean(kernel * f_vals, axis=0)  # (D,)


def pointwise_sobol_antiderivative(xy_batch, a, c, order, num_samples):
    return jax.vmap(lambda xy: monte_carlo_antiderivative_2d_flattened(
        x=xy[0], y=xy[1], a=a, c=c,
        order=order, num_samples=num_samples
    ))(xy_batch)


# def generate_sobol_sequences_2d(order, num_samples):
#     sobol_sequences = {}
#     for depth in range(1, order + 1):
#         sampler = qmc.Sobol(d=2, scramble=True)
#         samples = sampler.random_base2(m=int(np.log2(num_samples)))
#         sobol_sequences[depth] = jnp.array(samples)  # shape (num_samples, 2)
#     return sobol_sequences

# # Recursive nested MC in 2D using Sobol
# def monte_carlo_antiderivative_2d(x, y, a, c, order, num_samples, sobol_sequences):
#     out_dim = f(x, y).shape[0]
#     def recursive_antiderivative(current_order, x_val, y_val):
#         if current_order == 0:
#             return f(x_val, y_val)
#         else:
#             sobol = sobol_sequences[current_order]  # shape (num_samples, 2)
#             # Map [0,1]^2 → [a,x] × [c,y]
#             t_x = a + sobol[:, 0] * (x_val - a)
#             t_y = c + sobol[:, 1] * (y_val - c)

#             estimates = jnp.zeros((num_samples, out_dim))

#             def body_fun(i, acc):
#                 tx = t_x[i]
#                 ty = t_y[i]
#                 val = recursive_antiderivative(current_order - 1, tx, ty)
#                 return acc.at[i].set(val)

#             estimates = lax.fori_loop(0, num_samples, body_fun, estimates)
#             return (x_val - a) * (y_val - c) * jnp.mean(estimates, axis=0)

#     return recursive_antiderivative(order, x, y)



# def pointwise_sobol_antiderivative(xy_batch, a, c, order, num_samples, sobol_sequences):
#     def compute_fn(xy):
#         x, y = xy
#         return monte_carlo_antiderivative_2d(x, y, a, c, order, num_samples, sobol_sequences)
#     return jax.vmap(compute_fn)(xy_batch)


def build_2d_jax_sampler(image: jnp.ndarray):
    def sampler(xy_query: jnp.ndarray):
        return jax_interp2d(xy_query, image)
    return sampler

# ======================= Utility Functions =======================

def pad_image(image, pad_fraction=0.3):
    height, width = image.shape[:2]
    pad_h = int(height * pad_fraction)
    pad_w = int(width * pad_fraction)
    image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    return image


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


def _parse_args():
    parser = ArgumentParser("Signal Regression", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--activation", type=str, help="Activation function", default='swish')

    parser.add_argument("--num_channels", type=int, default=128, help="Number of channels in the MLP")

    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the MLP")

    parser.add_argument("--out_channel", type=int, default=3, help="Output Channel number")

    parser.add_argument("--in_channel", type=int, default=2, help="Input Channel number")

    parser.add_argument("--rotation", type=bool, default=False, help="Whether to use the Rotation Network")

    parser.add_argument('--learn_rate', type=float, default=1e-3, help="The network's learning rate")

    parser.add_argument('--schedule_step', type=int, default=5000, help="step to decrease the learning rate")

    parser.add_argument('--schedule_gamma', type=float, default=0.6, help="learning rate decrease factor")

    parser.add_argument("--pe", type=int, default=8, help="number of positional encoding functions")

    parser.add_argument("--num-steps", type=int, default=300000, help="Number of training steps.")

    parser.add_argument("--workers", type=int, default=12, help="number of workers")

    parser.add_argument("--batch", type=int, default=1024, help="Batch Size For training")

    parser.add_argument("--precision", type=int, default=32, help="Precision of the computation")

    parser.add_argument("--norm_exp", type=int, default=0, help="Normalization exponent")

    parser.add_argument("--experiment_name", type=str, default='experiment', help="experiment name")

    parser.add_argument("--norm_layer", type=str, default=None, help="Normalization layer")

    parser.add_argument("--summary", help="summary folder", default='')

    parser.add_argument("--image", type=str, default=None, help="Image path")

    parser.add_argument("--seed", type=int, default=100, help="seed value for the training")

    parser.add_argument("--order", type=int, default=1, help="order of integration")

    parser.add_argument("--samples", type=int, default=64, help="num samples")

    parser.add_argument("--analytic", type=str, default="none", help="type")

    return parser.parse_args()

def main():
    args = _parse_args()
    print("ARGS:", args, flush=True)
    print("Loading image:", args.image, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = random.PRNGKey(42)

    if args.precision == 32:    
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)
        print(f'--------------------- tensor type of computation : {args.precision} ----------------', flush=True)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)
        print(f'--------------------- tensor type of computation : {args.precision} ----------------', flush=True)

    pad = True
    loss_type = "l1"
    steps_till_summary = 100

    global f
    if args.analytic.lower() == "ackley":
        print("Using Ackley 2D")
        def f(x, y):
            return jnp.array([ackley_2d_jnp(x, y)])  # shape (1,)

    elif args.analytic.lower() == "gm":
        print("Using Gaussian Mixture 2D")
        gm_path = "../../data/analytic_params/gm_2d_params.npz"
        gm = GaussianMixture(gm_path)
        def f(x, y):
            return jnp.array([gm.eval(jnp.stack([x, y])[None])[0]])  # shape (1,)

    elif args.analytic.lower() == "hr":
        print("Using Hyperrectangle Mixture 2D")
        hr_path = "../../data/analytic_params/hr_2d_params.npz"
        hr = HyperrectangleMixture(hr_path)
        def f(x, y):
            return jnp.array([hr.eval(jnp.stack([x, y])[None])[0]])  # shape (1,)

    else:
        print("Using image-based interpolation")
        image = np.array(cv2.cvtColor(cv2.imread(args.image, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB))
        if pad:
            image = pad_envmap(image, 0.3)
        image_jnp = jnp.array(image)
        interpolator_fn = build_2d_jax_sampler(image_jnp)
        def f(x, y):
            return interpolator_fn(jnp.stack([x, y])[None])[0]  # shape (3,)

    if args.analytic.lower() in ["ackley", "gm", "hr"]:
        args.out_channel = 1


    # grid_res = 256
    # xx, yy = jnp.meshgrid(jnp.linspace(-1, 1, grid_res), jnp.linspace(-1, 1, grid_res))
    # xy_flat = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    # values = jax.vmap(lambda x: f(x[0], x[1]))(xy_flat).reshape(grid_res, grid_res)
    # values_np = np.array(values)
    # plt.figure(figsize=(6, 5))
    # plt.imshow(values_np, extent=[-1, 1, -1, 1], origin='lower', cmap=cm.viridis)
    # plt.colorbar(label="Function Value")
    # plt.title(f"2D Function Plot: {args.analytic}")
    # plt.axis("off")
    # plt.tight_layout() 
    # plt.savefig("seefunc2.png")

    model = CoordinateNet(args.out_channel,
                        args.activation,
                        args.in_channel,
                        args.num_channels,
                        args.num_layers,
                        args.pe,
                        True if args.norm_exp != 0 else False,
                        10,
                        norm_exp=args.norm_exp,
                        norm_layer=args.norm_layer).to(device)
    
    net_dictionary = dict(input=args.in_channel,
                        output=args.out_channel,
                        channels=args.num_channels,
                        layers=args.num_layers,
                        pe=True,
                        encodings=args.pe,
                        normalize_pe=True if args.norm_exp != 0 else False,
                        include_input=True,
                        activation=args.activation)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    model.train()
    writer = TrainingLog(args.experiment_name, add_unique_str=False)

    for step in range(args.num_steps):

                # Sample random query points in [-1, 1]^2
        st = time.time()
        key, subkey = random.split(key)
        model_input = jax.random.uniform(subkey, (args.batch, 2), minval=-1.0, maxval=1.0)  # (B, 2)

        gt = pointwise_sobol_antiderivative(model_input, a=-1.0, c=-1.0,
                                            order=args.order,
                                            num_samples=args.samples)


        model_input_torch = torch.from_numpy(np.asarray(model_input)).float().to(device)
        gt_torch = torch.from_numpy(np.asarray(gt)).float().to(device)

        pred = model(model_input_torch)
        loss = ((pred - gt_torch)**2).mean()



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        et = time.time()

        if step % 1000 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = step
            net_dictionary['optim'] = optimizer.state_dict()
            torch.save(net_dictionary, args.experiment_name + f'/checkpoint_{step}.pth')

        if step % steps_till_summary == 0:
            print(f"Step {step}, Loss: {loss.item():.9f}, Time: {et-st}", flush=True)
            writer.add_scalar('loss', loss.item(), step)


        if step % 100 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = step
            net_dictionary['optim'] = optimizer.state_dict()
            torch.save(net_dictionary, args.experiment_name + f'/current.pth')

    net_dictionary['ckpt'] = model.state_dict()
    net_dictionary['epoch'] = step
    net_dictionary['optim'] = optimizer.state_dict()
    torch.save(net_dictionary, args.experiment_name + f'/model_final.pth')


if __name__ == "__main__":
    main()
