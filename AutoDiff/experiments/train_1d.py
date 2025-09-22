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
import time
from utilities import ackley_1d, gaussian_mixture_1d, mixture_hyperrectangles


def map_range(values, old_range, new_range):
    new_width = (new_range[1] - new_range[0])
    old_width = (old_range[1] - old_range[0])
    return (((values - old_range[0]) * new_width) / old_width) + new_range[0]
    
def build_1d_sampler(data, method='linear'):
    x = np.linspace(0, data.shape[0] - 1, data.shape[0])
    return interp1d(x, data, kind=method, axis=0, bounds_error=False, fill_value='extrapolate')


def generate_training_samples_1d(batch_size, interpolator_fn, signal, precision=32):
    L = signal.shape[0] 
    random_samples_np = np.random.uniform(low=-100, high=100, size=[batch_size, 1])
    sample_coord = map_range(random_samples_np[:, 0], (-100, 100), (0, L - 1))
    
    input_tensor = torch.unsqueeze(torch.from_numpy(random_samples_np), 0).cuda()
    sampled_values = interpolator_fn(sample_coord)
    if len(sampled_values.shape) == 1:
        sampled_values = sampled_values.reshape(-1, 1)
    
    sampled_values = torch.from_numpy(sampled_values).cuda()
    signal_data = sampled_values.contiguous().view(-1, 1)  
    input_tensor = input_tensor.view(-1, 1)  
    input_tensor = input_tensor.float() if precision == 32 else input_tensor.double()
    signal_data = signal_data.float() if precision == 32 else signal_data.double()
    return input_tensor.cuda(), signal_data.cuda()


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
    loss_type = "l1"
    steps_till_summary = 200

    if args.analytic.lower() == "ackley":
        print("Using Ackley 1D function.")
        def analytic_fn(x): 
            return torch.from_numpy(ackley_1d(x.detach().cpu().numpy())).to(x.device).unsqueeze(-1).float()
        interpolator_fn = analytic_fn
        signal = None 
        input_range = (-1, 1)

    elif args.analytic.lower() == "gm":
        print("Using Gaussian Mixture 1D.")
        gm_pdf = gaussian_mixture_1d(seed=args.seed)
        def analytic_fn(x):  
            return torch.from_numpy(gm_pdf(x.detach().cpu().numpy())).to(x.device).unsqueeze(-1).float()
        interpolator_fn = analytic_fn
        signal = None
        input_range = (-1, 1)

    elif args.analytic.lower() == "hr":
        print("Using Hyperrectangle Mixture 1D.")
        def analytic_fn(x): 
            return torch.from_numpy(mixture_hyperrectangles(x.detach().cpu().numpy(), dim=1, seed=args.seed)).to(x.device).unsqueeze(-1).float()
        interpolator_fn = analytic_fn
        signal = None
        input_range = (-1, 1)

    else:
        print(f"Loading audio signal from {args.audio}")
        signal = load_mp3(args.audio)
        signal_name = os.path.basename(args.audio)
        print(f"Audio loaded: {signal_name}")
        plt.plot(signal)
        plt.show()
        if pad:
            signal = pad_signal_1d(signal, 0.3)
        interpolator_fn = build_1d_sampler(signal)
        input_range = (-100, 100)


    # plt.plot(interpolator_fn(torch.linspace(-1, 1, 2048)))
    # plt.savefig("seefunc.png")
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

        if step % 1000 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = step
            net_dictionary['optim'] = optim.state_dict()
            torch.save(net_dictionary, args.experiment_name + f'/current.pth')

        optim.zero_grad()

        if args.analytic.lower() in ["ackley", "gm", "hr"]:
            random_inputs = torch.FloatTensor(args.batch, 1).uniform_(*input_range).to(device)
            model_input = random_inputs.requires_grad_(True)
            ground_truth_f = interpolator_fn(model_input).reshape(-1, 1)
        else:
            model_input, ground_truth_f = generate_training_samples_1d(args.batch, interpolator_fn, signal)
        
        if args.order==1:
            derivative = vmap(jacfwd(lambda a: model(a)))(model_input).reshape(-1, 1)
        elif args.order==2:
            derivative = vmap(jacrev(jacfwd(lambda a: model(a))))(model_input).reshape(-1, 1)
        elif args.order==3:
            derivative = vmap(jacfwd(jacrev(jacfwd(lambda a: model(a)))))(model_input).reshape(-1, 1)

        f_derivative = derivative[:, :1]  # Single channel derivative

        if loss_type == "l1":
            loss_f = F.smooth_l1_loss(ground_truth_f, f_derivative)
        else:
            loss_f = ((f_derivative - ground_truth_f) ** 2).mean()
            
        loss = loss_f

        loss.backward()
        optim.step()
        scheduler.step()

        if step % steps_till_summary == 0:
            print(f'Iteration: {step}, Loss: {loss.item():.9f}', flush=True)
            writer.add_scalar('loss', loss_f.item(), step)

    
    net_dictionary['ckpt'] = model.state_dict()
    net_dictionary['epoch'] = step
    net_dictionary['optim'] = optim.state_dict()
    torch.save(net_dictionary, args.experiment_name + f'/model_final.pth')

    et = time.time()
    print(f"Total training time: {(et -st):.6f}", flush=True)


if __name__ == "__main__":
    main()