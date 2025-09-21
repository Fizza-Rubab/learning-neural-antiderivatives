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
import glob

def map_range(values, old_range, new_range):
    new_width = (new_range[1] - new_range[0])
    old_width = (old_range[1] - old_range[0])
    return (((values - old_range[0]) * new_width) / old_width) + new_range[0]
    
def build_1d_sampler(data, method='linear'):
    x = np.linspace(0, data.shape[0] - 1, data.shape[0])
    return interp1d(x, data, kind=method, axis=0, bounds_error=False, fill_value='extrapolate')

def generate_training_samples_1d(batch_size, interpolator_fn, signal, precision=32):
    L = signal.shape[0] 
    random_samples_np = np.random.uniform(low=-1, high=1, size=[batch_size, 1])
    sample_coord = map_range(random_samples_np[:, 0], (-1, 1), (0, L - 1))
    
    input_tensor = torch.unsqueeze(torch.from_numpy(random_samples_np), 0).cuda()
    sampled_values = interpolator_fn(sample_coord)
    if len(sampled_values.shape) == 1:
        sampled_values = sampled_values.reshape(-1, 1)
    
    sampled_values = torch.from_numpy(sampled_values).cuda()
    signal_data = sampled_values.contiguous().view(-1, 69)  

    input_tensor = input_tensor.view(-1, 1)  
    input_tensor = input_tensor.float() if precision == 32 else input_tensor.double()
    signal_data = signal_data.float() if precision == 32 else signal_data.double()
    return input_tensor.cuda(), signal_data.cuda()

def pad_signal_1d(signal, pad_fraction=0.3):
    length = signal.shape[0]
    pad_length = int(length * pad_fraction)
    pad_width = ((pad_length, pad_length), (0, 0), (0, 0))
    padded_signal = np.pad(signal, pad_width, mode='reflect')
    return padded_signal

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

    parser.add_argument("--pose", type=str, default=None, help="Path to pose file containing 1D signal")

    parser.add_argument("--seed", type=int, default=100, help="seed value for the training")

    parser.add_argument("--order", type=int, default=1, help="order of integration")


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
    
    raw_poses_path = args.pose
    signal = read_pose(raw_poses_path)
    steps_till_summary = 200
    print(f"Size before padding: ({signal.shape})", flush=True)

    interpolator_fn = build_1d_sampler(signal)

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

        model_input, ground_truth_f = generate_training_samples_1d(args.batch, interpolator_fn, signal)
        
        if args.order==1:
            derivative = vmap(jacfwd(lambda a: model(a)))(model_input).reshape(-1, 69)
        elif args.order==2:
            derivative = vmap(jacrev(jacfwd(lambda a: model(a))))(model_input).reshape(-1, 69)
        elif args.order==3:
            derivative = vmap(jacfwd(jacrev(jacfwd(lambda a: model(a)))))(model_input).reshape(-1, 69)

        # print("shapes", model_input.shape, ground_truth_f.shape, derivative.shape)
        f_derivative = derivative  

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