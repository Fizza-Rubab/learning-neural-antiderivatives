import os
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
import torch.nn.functional as F
import os
from scipy.interpolate import interp1d
import numpy as np
from torch.func import vmap, jacfwd, jacrev
import librosa
from model import CoordinateNet_ordinary as CoordinateNet
from utilities import calculate_psnr
from utilities import TrainingLog
import librosa
import matplotlib.pyplot as plt
from utilities import load_mp3
from utilities import ackley_1d_jnp, gaussian_mixture_1d_jnp, mixture_hyperrectangles_jnp,  GaussianMixture, HyperrectangleMixture
import time
import jax
import jax.numpy as jnp
from jax import random, lax
from scipy.stats import qmc
from torch.func import vmap, jacfwd, jacrev
import math
def nth_derivative(model, x, order):
    if order == 1:
        y = vmap(jacfwd(lambda a: model(a)))(x).reshape(-1, 1)
    elif order == 2:
        y = vmap(jacrev(jacfwd(lambda a: model(a))))(x).reshape(-1, 1)
    elif order == 3:
        y = vmap(jacfwd(jacrev(jacfwd(lambda a: model(a)))))(x).reshape(-1, 1)
    return y


def map_range(values, old_range, new_range):
    new_width = (new_range[1] - new_range[0])
    old_width = (old_range[1] - old_range[0])
    return (((values - old_range[0]) * new_width) / old_width) + new_range[0]
    

def pad_signal_1d(signal, pad_fraction=0.3):
    length = signal.shape[0]
    pad_length = int(length * pad_fraction)
    padded_signal = np.pad(signal.flatten(), (pad_length, pad_length), mode='reflect')
    return padded_signal[:, None]


def _parse_args():
    parser = ArgumentParser("Signal Regression", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--activation", type=str, help="Activation function", default='swish')

    parser.add_argument("--num_channels", type=int, default=128, help="Number of channels in the MLP")

    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the MLP")

    parser.add_argument("--out_channel", type=int, default=1, help="Output Channel number")

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

    parser.add_argument("--audio", type=str, default=None, help="Path to .npy file containing 1D signal")


    parser.add_argument("--seed", type=int, default=100, help="seed value for the training")

    parser.add_argument("--order", type=int, default=1, help="order of integration")

    parser.add_argument("--analytic", type=str, default="none", help="order of integration")

    parser.add_argument("--samples", type=int, default=64, help="no of samples")

    return parser.parse_args()



def load_audio_signal(audio_path):
    signal = np.load(audio_path)

    if signal.ndim > 1 and signal.shape[1] != 1:
        signal = signal.mean(axis=1)  
    signal = signal.flatten()
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-9)

    return signal.reshape(-1, 1)


def main():
    args = _parse_args()
    print(args, flush=True)
    print(f"Analytic mode: {args.analytic}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.precision == 32:
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)
        print(f'--------------------- tensor type of computation : {args.precision} ----------------')
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)
        print(f'--------------------- tensor type of computation : {args.precision} ----------------')

    pad = True
    loss_type = "l2"
    steps_till_summary = 100

    a = -1.0
    key = jax.random.PRNGKey(args.seed)

    if args.analytic.lower() == "ackley":
        print("Using Ackley 1D function.")
        def interpolator_fn(x):
            return jnp.squeeze(ackley_1d_jnp(x[None]))
    elif args.analytic.lower() == "gm":
        print("Using Gaussian Mixture 1D.")
        gm_path = "../../data/analytic_params/gm_1d_params.npz"
        gm = GaussianMixture(gm_path)
        def interpolator_fn(x):
            return jnp.array([gm.eval(jnp.array([[x]]))[0]])
    elif args.analytic.lower() == "hr":
        print("Using Hyperrectangle Mixture 1D.")
        hr_path = "../../data/analytic_params/hr_1d_params.npz"
        hr = HyperrectangleMixture(hr_path)
        def interpolator_fn(x):
            return jnp.array([hr.eval(jnp.array([[x]]))[0]])
    else:
        raise NotImplementedError("Only analytic modes ackley/gm/hr are supported here.")

    # def recursive_mc(x, a, order, num_samples):
    #     # def recursive(level, x_val):
    #     #     if level == 0:
    #     #         return interpolator_fn(x_val)
    #     #     sampler = qmc.Sobol(d=1, scramble=True)
    #     #     sobol = sampler.random_base2(m=int(np.log2(num_samples)))
    #     #     sobol = jnp.array(sobol[:, 0])
    #     #     t_samples = a + sobol * (x_val - a)
    #     #     vals = jax.vmap(lambda t: recursive(level - 1, t))(t_samples)
    #     #     return (x_val - a) * jnp.mean(vals)
    #     # return recursive(order, x)

    def flattened_mc(x, a, order, num_samples):
        factorial = math.factorial(order - 1)
        sampler = qmc.Sobol(d=1, scramble=True)
        sobol = sampler.random_base2(m=int(np.log2(num_samples)))
        sobol = jnp.array(sobol[:, 0])  # shape: (num_samples,)

        def estimate_at(x_val):
            t_samples = a + sobol * (x_val - a)
            f_vals = jax.vmap(interpolator_fn)(t_samples)
            kernel = ((x_val - t_samples) ** (order - 1)) / factorial
            return (x_val - a) * jnp.mean(kernel * f_vals)

        return estimate_at(x)


    # @jax.jit
    # def compute_gt(samples):
    #     return jax.vmap(lambda x: recursive_mc(x, a=a, order=args.order, num_samples=args.samples))(samples)

    @jax.jit
    def compute_gt(samples):
        return jax.vmap(lambda x: flattened_mc(x, a=a, order=args.order, num_samples=args.samples))(samples)


    plt.plot(interpolator_fn(jnp.linspace(-1, 1, 2048)))
    plt.savefig("seefunc.png")
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
    st = time.time()
    for step in range(args.num_steps):

        if step % 10000 == 0:
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
        key, subkey = random.split(key)
        random_samples = random.uniform(subkey, (args.batch,), minval=-1.0, maxval=1.0)
        model_input = torch.from_numpy(np.asarray(random_samples)).to(device).view(-1, 1).float()
        gt = compute_gt(random_samples)
        gt_torch = torch.from_numpy(np.asarray(gt)).to(device).view(-1, 1).float()
        
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

        
        # if step%10000==0:
        #     with torch.no_grad():
        #         eval_x = torch.linspace(-1, 1, 512).view(-1, 1).to(device).float()
        #         deriv_model = nth_derivative(model, eval_x, order=args.order).cpu().numpy()
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(deriv_model)
        #     plt.plot(interpolator_fn(jnp.linspace(-1, 1, 512)))
        #     plt.savefig(args.experiment_name + f'/{step}_results.png')


    
    net_dictionary['ckpt'] = model.state_dict()
    net_dictionary['epoch'] = step
    net_dictionary['optim'] = optim.state_dict()
    torch.save(net_dictionary, args.experiment_name + f'/model_final.pth')

    et = time.time()
    print(f"Total training time: {(et -st):.6f}", flush=True)


if __name__ == "__main__":
    main()