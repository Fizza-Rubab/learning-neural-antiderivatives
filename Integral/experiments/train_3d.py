import os
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
import torch.nn.functional as F
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from model import CoordinateNet_ordinary as CoordinateNet
from utilities import TrainingLog, mesh_to_sdf_tensor
import time
from scipy.stats import qmc
from utilities import ackley_3d_jnp, gaussian_mixture_3d_jnp, mixture_hyperrectangles_jnp, GaussianMixture, HyperrectangleMixture
import math
import matplotlib.pyplot as plt

def jax_interp3d(xyz_query, volume):
    D, H, W = volume.shape
    x = jnp.clip((xyz_query[:, 0] + 1.0) * 0.5 * (W - 1), 0, W - 2)
    y = jnp.clip((xyz_query[:, 1] + 1.0) * 0.5 * (H - 1), 0, H - 2)
    z = jnp.clip((xyz_query[:, 2] + 1.0) * 0.5 * (D - 1), 0, D - 2)

    x0, y0, z0 = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32), jnp.floor(z).astype(jnp.int32)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    xd, yd, zd = x - x0, y - y0, z - z0

    def get_voxel(ix, iy, iz):
        return volume[iz, iy, ix]  # Note: z, y, x indexing

    c000 = jax.vmap(get_voxel)(x0, y0, z0)
    c001 = jax.vmap(get_voxel)(x0, y0, z1)
    c010 = jax.vmap(get_voxel)(x0, y1, z0)
    c011 = jax.vmap(get_voxel)(x0, y1, z1)
    c100 = jax.vmap(get_voxel)(x1, y0, z0)
    c101 = jax.vmap(get_voxel)(x1, y0, z1)
    c110 = jax.vmap(get_voxel)(x1, y1, z0)
    c111 = jax.vmap(get_voxel)(x1, y1, z1)

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    return c0 * (1 - zd) + c1 * zd

def build_3d_jax_sampler(volume):
    def sampler(xyz_query):
        return jax_interp3d(xyz_query, volume)
    return sampler

# def generate_sobol_sequences_3d(order, num_samples):
#     sobol_sequences = {}
#     for depth in range(1, order + 1):
#         sampler = qmc.Sobol(d=3, scramble=True)
#         samples = sampler.random_base2(m=int(np.log2(num_samples)))
#         sobol_sequences[depth] = jnp.array(samples)
#     return sobol_sequences

# def monte_carlo_antiderivative_3d(x, y, z, a, c, e, order, num_samples, sobol_sequences, f):
#     def recursive(current_order, x_val, y_val, z_val):
#         if current_order == 0:
#             return f(x_val, y_val, z_val)

#         sobol = sobol_sequences[current_order]
#         t_x = a + sobol[:, 0] * (x_val - a)
#         t_y = c + sobol[:, 1] * (y_val - c)
#         t_z = e + sobol[:, 2] * (z_val - e)

#         sample_val = f(x_val, y_val, z_val)
#         out_dim = sample_val.shape[0] if len(sample_val.shape) > 0 else 1
#         estimates = jnp.zeros((num_samples, out_dim))

#         def body_fun(i, acc):
#             tx, ty, tz = t_x[i], t_y[i], t_z[i]
#             val = recursive(current_order - 1, tx, ty, tz)
#             return acc.at[i].set(val)

#         estimates = jax.lax.fori_loop(0, num_samples, body_fun, estimates)
#         return (x_val - a) * (y_val - c) * (z_val - e) * jnp.mean(estimates, axis=0)

#     return recursive(order, x, y, z)



# def pointwise_sobol_antiderivative_3d(xyz_batch, a, c, e, order, num_samples, sobol_sequences, f):
#     def compute_fn(xyz):
#         return monte_carlo_antiderivative_3d(xyz[0], xyz[1], xyz[2], a, c, e, order, num_samples, sobol_sequences, f)
#     return jax.vmap(compute_fn)(xyz_batch)

def monte_carlo_antiderivative_3d_flattened(x, y, z, a, c, e, order, num_samples, f):
    factorial = math.factorial(order - 1)

    # Generate Sobol samples in [0, 1]^3
    sampler = qmc.Sobol(d=3, scramble=True)
    sobol = sampler.random_base2(m=int(np.log2(num_samples)))
    sobol_samples = jnp.array(sobol)  # shape (num_samples, 3)

    # Rescale to [a,x], [c,y], [e,z]
    t_x = a + sobol_samples[:, 0] * (x - a)
    t_y = c + sobol_samples[:, 1] * (y - c)
    t_z = e + sobol_samples[:, 2] * (z - e)

    # Evaluate f at these samples
    t_xyz = jnp.stack([t_x, t_y, t_z], axis=1)  # shape (num_samples, 3)
    f_vals = f(t_xyz)  # shape (num_samples, D)


    kernel_x = (x - t_x) ** (order - 1)
    kernel_y = (y - t_y) ** (order - 1)
    kernel_z = (z - t_z) ** (order - 1)
    kernel = (kernel_x * kernel_y * kernel_z)[:, None] / (factorial ** 3)  # shape (num_samples, 1)

    return (x - a) * (y - c) * (z - e) * jnp.mean(kernel * f_vals, axis=0)  # shape (D,)

def pointwise_sobol_antiderivative_3d(xyz_batch, a, c, e, order, num_samples, f):
    return jax.vmap(lambda xyz: monte_carlo_antiderivative_3d_flattened(
        x=xyz[0], y=xyz[1], z=xyz[2],
        a=a, c=c, e=e,
        order=order, num_samples=num_samples, f=f
    ))(xyz_batch)


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

def _parse_args():
    parser = ArgumentParser("Signal Regression", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--activation", type=str, help="Activation function", default='swish')

    parser.add_argument("--num_channels", type=int, default=128, help="Number of channels in the MLP")

    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the MLP")

    parser.add_argument("--out_channel", type=int, default=1, help="Output Channel number")

    parser.add_argument("--in_channel", type=int, default=3, help="Input Channel number")

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

    parser.add_argument("--object", type=str, default=None, help="Object model path")

    parser.add_argument("--seed", type=int, default=100, help="seed value for the training")
    
    parser.add_argument("--order", type=int, default=100, help="order of integration")

    parser.add_argument("--samples", type=int, default=128, help="order of integration")

    parser.add_argument("--analytic", type=str, default="none", help="type")

    return parser.parse_args()

def main():
    args = _parse_args()
    print(args, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print("Using Ackley 3D")
        def f(xyz):  # xyz: (N, 3)
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            return ackley_3d_jnp(x, y, z)[..., None]  # (N, 1)

    elif args.analytic.lower() == "gm":
        print("Using Gaussian Mixture 3D")
        gm_path = "../../data/analytic_params/gm_3d_params.npz"
        gm = GaussianMixture(gm_path)
        def f(xyz):
            return gm.eval(xyz)[..., None]  # shape (N, 1)

    elif args.analytic.lower() == "hr":
        print("Using Hyperrectangle Mixture 3D")
        hr_path = "../../data/analytic_params/hr_3d_params.npz"
        hr = HyperrectangleMixture(hr_path)
        def f(xyz):
            return hr.eval(xyz)[..., None]  # shape (N, 1)

    else:
        sdf = mesh_to_sdf_tensor(args.object, 256)
        print(f"Size before padding: ({sdf.shape})", flush=True)
        if pad:
            sdf = pad_sdf(sdf, 0.3)
        print(f"Size after padding: ({sdf.shape})", flush=True)
        interpolator_fn = build_3d_jax_sampler(jnp.array(sdf))
        def f(xyz):  # xyz: (N, 3)
            return interpolator_fn(xyz)[..., None]  # shape (N, 1)

    # res = 64
    # grid = jnp.linspace(-1, 1, res)
    # xx, yy = jnp.meshgrid(grid, grid, indexing='ij')
    # z_center = grid[res // 2]
    # coords = jnp.stack([xx.ravel(), yy.ravel(), jnp.full(xx.size, z_center)], axis=-1)
    # values = jnp.array([f(*pt) for pt in coords])
    # Z = values.reshape(res, res)
    # plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    # plt.title("Z-slice at center")
    # plt.colorbar()
    # plt.savefig("seefunc3.png")
    
    plt.show()
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

    print("No. of parameters", sum(p.numel() for p in model.parameters()), flush=True)

    optim = torch.optim.Adam(lr=args.learn_rate, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.schedule_step, gamma=args.schedule_gamma)
    writer = TrainingLog(args.experiment_name, add_unique_str=False)

    net_dictionary = dict(input=args.in_channel,
                        output=args.out_channel,
                        channels=args.num_channels,
                        layers=args.num_layers,
                        pe=True,
                        encodings=args.pe,
                        normalize_pe=True if args.norm_exp != 0 else False,
                        include_input=True,
                        activation=args.activation)


    key = random.PRNGKey(args.seed)

    for step in range(args.num_steps):
        st = time.time()
        key, subkey = jax.random.split(key)
        model_input = jax.random.uniform(subkey, (args.batch, 3), minval=-1.0, maxval=1.0)

        gt = pointwise_sobol_antiderivative_3d(model_input, a=-1.0, c=-1.0, e=-1.0,
                                       order=args.order,
                                       num_samples=args.samples,
                                       f=f).reshape(-1, args.out_channel)



        model_input_torch = torch.from_numpy(np.asarray(model_input)).float().to(device)
        gt_torch = torch.from_numpy(np.asarray(gt)).float().to(device)

        # Predict
        pred = model(model_input_torch)


        loss = ((pred - gt_torch) ** 2).mean()


        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()    
        et = time.time()
        
        if step % 1000 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = step
            net_dictionary['optim'] = optim.state_dict()
            torch.save(net_dictionary, args.experiment_name + f'/checkpoint_{step}.pth')

        if step % steps_till_summary == 0:
            print(f"Step {step}, Loss: {loss.item():.9f}, Time:{et - st}", flush=True)
            writer.add_scalar('loss', loss.item(), step)


        if step % 100 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = step
            net_dictionary['optim'] = optim.state_dict()
            torch.save(net_dictionary, args.experiment_name + f'/current.pth')

    net_dictionary['ckpt'] = model.state_dict()
    net_dictionary['epoch'] = step
    net_dictionary['optim'] = optim.state_dict()
    torch.save(net_dictionary, args.experiment_name + f'/model_final.pth')


if __name__ == "__main__":
    main()
