import os
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.func import vmap, jacfwd, jacrev
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
import torch.nn.functional as F
import numpy as np
import time
import jax
import jax.numpy as jnp
from jax import random, lax
from scipy.stats import qmc
from model import CoordinateNet_ordinary as CoordinateNet
from utilities import calculate_psnr, TrainingLog
import math
# ======================== Integration & Sampling ========================


def interpolate_vector_1d(x, x_vals, y_vals):
    idx = jnp.clip(jnp.searchsorted(x_vals, x, side="right") - 1, 0, len(x_vals) - 2)
    x0 = x_vals[idx]
    x1 = x_vals[idx + 1]
    y0 = y_vals[idx]
    y1 = y_vals[idx + 1]
    weight = (x - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)


def nth_derivative(model, x, order):
    if order == 1:
        y = vmap(jacfwd(lambda a: model(a)))(x).reshape(-1, 1)
    elif order == 2:
        y = vmap(jacrev(jacfwd(lambda a: model(a))))(x).reshape(-1, 1)
    elif order == 3:
        y = vmap(jacfwd(jacrev(jacfwd(lambda a: model(a)))))(x).reshape(-1, 1)
    return y



def monte_carlo_antiderivative(x, a, order, num_samples, f_array, x_vals):
    factorial = math.factorial(order - 1)
    sampler = qmc.Sobol(d=1, scramble=True)
    sobol = sampler.random_base2(m=int(np.log2(num_samples)))
    sobol_samples = jnp.array(sobol[:, 0])  # shape (num_samples,)

    def estimate_at(x_val):
        t_samples = a + sobol_samples * (x_val - a)
        def interp_fn(t):
            return interpolate_vector_1d(t, x_vals, f_array)  # returns (D,)
        
        f_vals = jax.vmap(interp_fn)(t_samples)  # shape (num_samples, D)
        kernel = ((x_val - t_samples) ** (order - 1))[:, None] / factorial  # shape (num_samples, 1)
        estimate = (x_val - a) * jnp.mean(kernel * f_vals, axis=0)  # shape (D,)
        return estimate
    return estimate_at(x)


# ======================== Data Utils ========================

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
    if num_frames < total_frames:
        start = (total_frames - num_frames) // 2
        pose_array = pose_array[start:start + num_frames]

    if normalize:
        min_val, max_val = pose_array.min(), pose_array.max()
        pose_array = (pose_array - min_val) / (max_val - min_val)
    return pose_array

def _parse_args():
    parser = ArgumentParser("Signal Regression", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--activation", type=str, help="Activation function", default='swish')

    parser.add_argument("--num_channels", type=int, default=128, help="Number of channels in the MLP")

    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the MLP")

    parser.add_argument("--out_channel", type=int, default=69, help="Output Channel number")

    parser.add_argument("--in_channel", type=int, default=1, help="Input Channel number")

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

    parser.add_argument("--pose", type=str, default=None, help="Path to .npy file containing 1D signal")


    parser.add_argument("--seed", type=int, default=100, help="seed value for the training")

    parser.add_argument("--order", type=int, default=1, help="order of integration")

    parser.add_argument("--samples", type=int, default=1024, help="number of samples")

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

    motion_data = read_pose(args.pose)
    signal = motion_data.reshape(motion_data.shape[0], -1)  # shape: (T, 69)

    steps_till_summary = 100
    print(f"Size before padding: ({signal.shape})", flush=True)
    # if pad:
    #     signal = pad_signal_1d(signal, 0.3)
    print(f"Size after padding: ({signal.shape})", flush=True)

    # Setup for antiderivative training
    T = signal.shape[0]
    x_vals = jnp.linspace(-1.0, 1.0, T)
    f_array = jnp.array(signal)  # shape (T, D)
    a = -1.0  # integration lower bound
    key = jax.random.PRNGKey(args.seed)


    global interpolator_fn
    interpolator_fn = interpolate_vector_1d

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
    model = model.double() if args.precision == 64 else model.float()
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.schedule_step, gamma=args.schedule_gamma)

    net_dictionary = dict(input=args.in_channel,
                        output=args.out_channel,
                        channels=args.num_channels,
                        layers=args.num_layers,
                        pe=True,
                        encodings=args.pe,
                        normalize_pe=True if args.norm_exp != 0 else False,
                        include_input=True,
                        activation=args.activation)

    writer = TrainingLog(args.experiment_name, add_unique_str=False)
    for step in range(args.num_steps):

        if step % 1000 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = step
            net_dictionary['optim'] = optim.state_dict()
            torch.save(net_dictionary, args.experiment_name + f'/checkpoint_{step}.pth')

        if step % 100 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = step
            net_dictionary['optim'] = optim.state_dict()
            torch.save(net_dictionary, args.experiment_name + f'/current.pth')
        st = time.time()

        optim.zero_grad()
        subkey, key = random.split(key)
        random_samples = random.uniform(subkey, (args.batch,), minval=-1.0, maxval=1.0)

        @jax.jit
        def compute_gt(samples):
            return jax.vmap(lambda x: monte_carlo_antiderivative(
                x=x,
                a=a,
                order=args.order,
                num_samples=args.samples,
                f_array=f_array,
                x_vals=x_vals))(samples)


        gt = compute_gt(random_samples)

        model_input = torch.from_numpy(np.asarray(random_samples)).float().to(device).view(-1, 1)
        gt_torch = torch.from_numpy(np.asarray(gt)).float().to(device)

        pred = model(model_input)

        loss_f = ((pred - gt_torch) ** 2).mean()
            
        loss = loss_f

        loss.backward()
        optim.step()
        scheduler.step()
        et = time.time()
        if step % steps_till_summary == 0:
            print(f'Iteration: {step}, Loss: {loss.item():.9f}, Time: {et-st}', flush=True)
            writer.add_scalar('loss', loss_f.item(), step)

        # if step%1000==0:
        #     with torch.no_grad():
        #         eval_x = torch.linspace(-1, 1, 512).view(-1, 1).to(device).float()
        #         deriv_model = nth_derivative(model, eval_x, order=args.order).cpu().numpy()
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(deriv_model)
        #     plt.plot(interpolator_fn(jnp.linspace(-1, 1, 512)))
        #     plt.savefig(f'{step}_results.png')
            


    
    net_dictionary['ckpt'] = model.state_dict()
    net_dictionary['epoch'] = step
    net_dictionary['optim'] = optim.state_dict()
    torch.save(net_dictionary, args.experiment_name + f'/model_final.pth')

    et = time.time()
    print(f"Total training time: {(et -st):.6f}", flush=True)


if __name__ == "__main__":
    main()      