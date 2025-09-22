import os
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
import torch.nn.functional as F
import os
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from torch.func import vmap, jacfwd, jacrev
import cv2
from model import CoordinateNet_ordinary as CoordinateNet
from utilities import calculate_psnr
from utilities import TrainingLog
import time
from utilities import ackley_2d, gaussian_mixture_2d, mixture_hyperrectangles


def map_range(values, old_range, new_range):
    new_width = (new_range[1] - new_range[0])
    old_width = (old_range[1] - old_range[0])
    return (((values - old_range[0]) * new_width) / old_width) + new_range[0]
    
def build_2d_sampler(data, method='linear'):
    x = np.linspace(0, data.shape[0] - 1, data.shape[0])
    y = np.linspace(0, data.shape[1] - 1, data.shape[1])
    return RegularGridInterpolator((x, y), data, method=method)

def generate_training_samples_2d(batch_size, interpolator_fn, img, precision=32):
    H, W = img.shape[:2]
    random_samples_np = np.random.uniform(low=-1, high=1, size=[batch_size, 2])
    sample_coord_x = map_range(random_samples_np[:, 0], (-1, 1), (0, H - 1)).reshape(-1,1)
    sample_coord_y = map_range(random_samples_np[:, 1], (-1, 1), (0, W - 1)).reshape(-1,1)
    sample_coord =  np.concatenate([sample_coord_x, sample_coord_y], axis=1)
    input_tensor = torch.unsqueeze(torch.from_numpy(random_samples_np), 0).cuda()
    bi_sampled = interpolator_fn(sample_coord)
    bi_sampled = torch.from_numpy(bi_sampled).cuda()
    rgb_data = bi_sampled.contiguous().view(-1, 3)
    input_tensor = input_tensor.view(-1, 2)
    input_tensor = input_tensor.float() if precision == 32 else input_tensor.double()
    rgb_data = rgb_data.float() if precision == 32 else rgb_data.double()
    return input_tensor.cuda(), rgb_data.cuda()

def pad_image(image, pad_fraction=0.3):
    height, width = image.shape[:2]
    pad_height = int(height * pad_fraction)
    pad_width = int(width * pad_fraction)
    padded_image = np.pad(image, ((0, 0), (pad_width, pad_width), (0,0)), mode='reflect')
    padded_image = np.pad(padded_image, ((pad_height, pad_height), (0, 0), (0,0)), mode='reflect')
    return padded_image


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

    parser.add_argument("--analytic", type=str, default="none", help="order of integration")

    return parser.parse_args()


def main():
    args = _parse_args()
    print("ARGS:", args, flush=True)
    print("Loading image:", args.image, flush=True)


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
    steps_till_summary = 200
    input_range = (-1, 1)
    if args.analytic.lower() == "ackley":
        print("Using 2D Ackley function")
        def analytic_fn(xy):
            x, y = xy[:, 0].detach().cpu().numpy(), xy[:, 1].detach().cpu().numpy()
            val = ackley_2d(x, y)
            return torch.from_numpy(val).to(xy.device).unsqueeze(-1).float()
        interpolator_fn = analytic_fn
        img = None
        args.out_channel = 1

    elif args.analytic.lower() == "gm":
        print("Using 2D Gaussian Mixture function")
        gm_pdf = gaussian_mixture_2d(seed=args.seed)
        def analytic_fn(xy):
            val = gm_pdf(xy.detach().cpu().numpy())
            return torch.from_numpy(val).to(xy.device).unsqueeze(-1).float()
        interpolator_fn = analytic_fn
        img = None
        args.out_channel = 1


    elif args.analytic.lower() == "hr":
        print("Using 2D Hyperrectangle Mixture function")
        def analytic_fn(xy):
            val = mixture_hyperrectangles(xy.detach().cpu().numpy(), dim=2, seed=args.seed, num_rects=5, rotation=True)
            return torch.from_numpy(val).to(xy.device).unsqueeze(-1).float()
        interpolator_fn = analytic_fn
        img = None
        args.out_channel = 1

    else:
        print(f"Loading image from {args.image}")
        img_name = os.path.basename(args.image)
        img = np.array(cv2.cvtColor(cv2.imread(args.image, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)) / 255.0
        print(f"Size before padding: ({img.shape})", flush=True)
        if pad:
            img = pad_image(img, 0.3)
        print(f"Size after padding: ({img.shape})", flush=True)

        interpolator_fn = build_2d_sampler(img)
        input_range = (-1, 1)


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
            random_inputs = torch.FloatTensor(args.batch, 2).uniform_(*input_range).to(device)
            model_input = random_inputs.requires_grad_(True)
            ground_truth_f = analytic_fn(model_input)
        else:
            model_input, ground_truth_f = generate_training_samples_2d(args.batch, interpolator_fn, img)

        if args.order == 1:
            derivative = vmap(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1))(
                model_input[:, 0:1], model_input[:, 1:2]
            ).reshape(-1, args.out_channel)
        elif args.order == 2:
            derivative = vmap(jacrev(jacfwd(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1), argnums=0), argnums=1))(
                model_input[:, 0:1], model_input[:, 1:2]
            ).reshape(-1, args.out_channel)
        
        f_derivative = derivative[:, :3]

        if loss_type == "l1":
            loss_f = F.smooth_l1_loss(ground_truth_f, f_derivative)
        else:
            loss_f = ((f_derivative - ground_truth_f) ** 2).mean()
            
        loss = loss_f

        loss.backward()
        optim.step()
        scheduler.step()

        if  step % steps_till_summary == 0:
            print(f'Iteration: {step}, Loss: {loss.item():.6f}', flush=True)
            writer.add_scalar('loss', loss_f.item(), step)

    net_dictionary['ckpt'] = model.state_dict()
    net_dictionary['epoch'] = step
    net_dictionary['optim'] = optim.state_dict()
    torch.save(net_dictionary, args.experiment_name + f'/model_final.pth')

    et = time.time()
    print(f"Total training time: {(et -st):.6f}", flush=True)


if __name__ == "__main__":
    main()