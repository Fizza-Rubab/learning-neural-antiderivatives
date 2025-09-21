import numpy as np
import torch
from .utils import map_range
from datetime import datetime
import os
import torch.utils.tensorboard as tb


def now_string():
   return datetime.utcnow().strftime("%Y.%m.%d_%H.%M.%S.%f")[:-3]


# Log training progress to tensorboard
class TrainingLog:

    def __init__(self, log_dir, add_unique_str=True):
        if add_unique_str:
            log_dir = os.path.join(log_dir, now_string())
        self.writer = tb.SummaryWriter(log_dir)

    # make sure that data is sent to tensorboard
    def flush(self):
        self.writer.flush()

    # close the logger
    def close(self):
        self.writer.close()

    # add a scalar
    def add_scalar(self, name, value, step, walltime=None):
        self.writer.add_scalar(name, value, global_step=step, walltime=walltime)

    # add an image
    def add_image(self, name, image, step=0, force_flush=True):
        self.writer.add_image(name, image, step, dataformats='HWC')
        if force_flush:
            self.flush()

    # add a graph. (The graph can be a neural network)
    def add_graph(self, graph, graph_input):
        self.writer.add_graph(graph, graph_input)


def generate_training_samples_1d(args, interpolator_fn, signal_np, full=False):
    R = 1
    if signal_np is not None:
        L = signal_np.shape[0]
        if full:
            coords = np.linspace(-R, R, L).reshape(-1, 1)
        else:
            coords = np.random.uniform(low=-R, high=R, size=[args.batch, 1]) * ((L - 1) / L)

        sample_coord = map_range(coords, (-R, R), (0, L - 1))
        input_tensor = torch.from_numpy(coords).cuda().view(-1, args.in_channel)

        sampled = np.asarray(interpolator_fn(sample_coord))
        sampled = sampled.reshape(-1, args.out_channel) if sampled.ndim == 1 else sampled
        output_tensor = torch.from_numpy(sampled).cuda()
        input_tensor = input_tensor.float() if args.precision == 32 else input_tensor.double()
        output_tensor = output_tensor.float() if args.precision == 32 else output_tensor.double()

        return input_tensor, output_tensor
    else:
        input_tensor = torch.FloatTensor(args.batch, 1).uniform_(-1, 1).cuda()
        input_tensor = input_tensor.float() if args.precision == 32 else input_tensor.double()
        output_tensor = interpolator_fn(input_tensor).view(-1, args.out_channel).cuda()
        output_tensor = output_tensor.float() if args.precision == 32 else output_tensor.double()
        return input_tensor, output_tensor



def generate_training_samples_1d_motion(args, interpolator_fn, signal_np, full=False):
    L = signal_np.shape[0]  
    if full:
        coords = np.linspace(-1, 1, L).reshape(-1, 1)
    else:
        coords = np.random.uniform(low=-1, high=1, size=[args.batch, 1]) * ((L - 1) / L)

    sample_coord = map_range(coords, (-1, 1), (0, L - 1))
    input_tensor = torch.from_numpy(coords).cuda().view(-1, args.in_channel)
    sampled = np.asarray(interpolator_fn(sample_coord))
    sampled = sampled.reshape(-1, args.out_channel) if sampled.ndim == 1 else sampled
    output_tensor = torch.from_numpy(sampled).cuda()
    input_tensor = input_tensor.float() if args.precision == 32 else input_tensor.double()
    output_tensor = output_tensor.float() if args.precision == 32 else output_tensor.double()
    return input_tensor, output_tensor


# def generate_training_samples_2d(args, interpolator_fn, monte_carlo_np):
#     H = monte_carlo_np.shape[0]

#     random_samples_np = np.random.uniform(low=-1, high=1, size=[args.batch, 2]) * ((H - 1) / H)
#     sample_coord = map_range(random_samples_np, (-1, 1), (0, H - 1))

#     input_tensor = torch.unsqueeze(torch.from_numpy(random_samples_np), 0).cuda()
#     bi_sampled = interpolator_fn(sample_coord)

#     bi_sampled = torch.from_numpy(bi_sampled).cuda()
#     rgb_data = bi_sampled.contiguous().view(-1, args.out_channel)

#     input_tensor = input_tensor.view(-1, args.in_channel)
#     input_tensor = input_tensor.float() if args.precision == 32 else input_tensor.double()
#     monte_carlo_rgb = rgb_data.float() if args.precision == 32 else rgb_data.double()

#     return input_tensor, monte_carlo_rgb

def generate_training_samples_2d(args, interpolator_fn, monte_carlo_np, full=False):

    if monte_carlo_np is not None:
        H = monte_carlo_np.shape[0]

        if full:
            x = np.linspace(-1, 1, H)
            coords = np.stack(np.meshgrid(x, x), -1).reshape(-1, 2)
        else:
            coords = np.random.uniform(low=-1, high=1, size=[args.batch, 2]) * ((H - 1) / H)

        sample_coord = map_range(coords, (-1, 1), (0, H - 1))
        input_tensor = torch.from_numpy(coords).cuda().unsqueeze(0).view(-1, args.in_channel)
        bi_sampled = interpolator_fn(sample_coord)

        bi_sampled = torch.from_numpy(bi_sampled).cuda()
        rgb_data = bi_sampled.contiguous().view(-1, args.out_channel)

        input_tensor = input_tensor.float() if args.precision == 32 else input_tensor.double()
        monte_carlo_rgb = rgb_data.float() if args.precision == 32 else rgb_data.double()

        return input_tensor, monte_carlo_rgb
    else:
        input_tensor = torch.FloatTensor(args.batch, 2).uniform_(-1, 1).cuda()
        input_tensor = input_tensor.float() if args.precision == 32 else input_tensor.double()
        output_tensor = interpolator_fn(input_tensor).view(-1, args.out_channel).cuda()
        output_tensor = output_tensor.float() if args.precision == 32 else output_tensor.double()
        return input_tensor, output_tensor

def generate_training_samples_3d(args, interpolator_fn, monte_carlo_np):
    if monte_carlo_np is not None:
        random_samples_np = np.random.uniform(low=-1, high=1, size=[args.batch, 3])
        tri_sampled = interpolator_fn(random_samples_np)
        tri_sampled = torch.from_numpy(tri_sampled).cuda()

        rand_torch = torch.unsqueeze(torch.from_numpy(random_samples_np), 0).cuda()
        input_tensor = rand_torch.view(-1, 3)

        monte_gt = tri_sampled.contiguous().view(-1, 1)
        input_tensor = input_tensor.view(-1, 3)

        monte_gt = monte_gt.float() if args.precision == 32 else monte_gt.double()
        input_tensor = input_tensor.float() if args.precision == 32 else input_tensor.double()

        return input_tensor, monte_gt
    else:
        input_tensor = torch.FloatTensor(args.batch, 3).uniform_(-1, 1).cuda()
        input_tensor = input_tensor.float() if args.precision == 32 else input_tensor.double()
        output_tensor = interpolator_fn(input_tensor).view(-1, args.out_channel).cuda()
        output_tensor = output_tensor.float() if args.precision == 32 else output_tensor.double()
        return input_tensor, output_tensor


def generate_training_samples_video(args, trilinear_interpolation, monte_carlo_np):
    random_samples_np = np.random.uniform(low=-1, high=1, size=[args.batch, 3])
    tri_sampled = trilinear_interpolation(random_samples_np)
    tri_sampled = torch.from_numpy(tri_sampled).cuda()

    rand_torch = torch.unsqueeze(torch.from_numpy(random_samples_np), 0).cuda()
    input_tensor = rand_torch.view(-1, 3)

    rgb_data = tri_sampled.contiguous().view(-1, 3)
    input_tensor = input_tensor.view(-1, 3)

    rgb_data = rgb_data.float() if args.precision == 32 else rgb_data.double()
    input_tensor = input_tensor.float() if args.precision == 32 else input_tensor.double()
    monte_carlo_rgb = rgb_data

    return input_tensor, monte_carlo_rgb
