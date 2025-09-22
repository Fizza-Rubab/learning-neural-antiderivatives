import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import CoordinateNet_ordinary as CoordinateNet
from torch.func import vmap, jacfwd, jacrev
from utilities import ackley_3d, gaussian_mixture_3d, mixture_hyperrectangles
import time


def build_3d_grid(D, H, W):
    zs = torch.linspace(-1, 1, D)
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack([x, y, z], dim=-1)
    return grid.view(-1, 3).cuda()


def get_ground_truth_3d(func_name, coords):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    if func_name == "ackley":
        return ackley_3d(x, y, z)
    elif func_name == "gm":
        return gaussian_mixture_3d(seed=100)(coords)
    elif func_name == "hr":
        return mixture_hyperrectangles(coords, dim=3, seed=100, num_rects=45)
    else:
        raise ValueError(f"Unknown function: {func_name}")


def nth_derivative(model, x, order):
    if order == 0:
        return vmap(jacfwd(jacrev(jacfwd(lambda a, b, c: model(torch.cat([a, b, c], -1)), argnums=0), argnums=1), argnums=2))(
            x[:, 0:1], x[:, 1:2], x[:, 2:3]
        ).reshape(-1, 1)
    elif order == 1:
        return vmap(jacfwd(jacrev(jacfwd(jacfwd(jacrev(jacfwd(lambda a, b, c: model(torch.cat([a, b, c], -1)), argnums=0), argnums=1), argnums=2), argnums=0), argnums=1), argnums=2))(
                x[:, 0:1], x[:, 1:2], x[:, 2:3]
            ).reshape(-1, 1)
    else:
        raise ValueError("Only orders 0 and 1 are supported in this evaluation.")


def chunked_derivative(model, coords, order, chunk_size=4096):
    outputs = []
    x = 1
    for i in range(0, coords.shape[0], chunk_size):
        print(x)
        chunk = coords[i:i+chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
        x+=1
    return torch.cat(outputs, dim=0).numpy()


def plot_sdf_slice(pred, gt, z_idx=None, save_path="sdf_slice.png"):
    if z_idx is None:
        z_idx = pred.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pred[z_idx], cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title("Predicted")
    axes[1].imshow(gt[z_idx], cmap='seismic', vmin=-1, vmax=1)
    axes[1].set_title("Ground Truth")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    func_names = [ "gm", "hr", "ackley"]
    size = 64
    orders = [0, 1]
    chunk_size = 4096
    ckpt_root = "../models/FD-Blur/3d"
    plot_dir = "eval_3d_analytic_plots"
    os.makedirs(plot_dir, exist_ok=True)

    results_log = [] 
    for func_name in func_names:
        for order in orders:
            ckpt_path = os.path.join(ckpt_root, f"{func_name}_order={order}.pth")
            print(func_name, order)
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}")
                continue

            weights = torch.load(ckpt_path)
            model = CoordinateNet(
                weights['output'], weights['activation'], weights['input'],
                weights['channels'], weights['layers'], weights['encodings'],
                weights['normalize_pe'], weights['pe'], norm_exp=0
            ).cuda()
            model.load_state_dict(weights['ckpt'])
            model.eval()

            coords = build_3d_grid(size, size, size)
            
            pred = chunked_derivative(model, coords, order, chunk_size).reshape(size, size, size)

            if order==1:
                pred*=-1
            x_np = coords.detach().cpu().numpy()
            gt = get_ground_truth_3d(func_name, x_np).reshape(size, size, size)

            mse = np.mean((pred - gt) ** 2)
            line = f"{func_name}, order={order}, MSE={mse:.8f}"
            print(line)
            results_log.append(line)

            slice_path = os.path.join(plot_dir, f"{func_name}_order{order}.png")
            plot_sdf_slice(pred, gt, save_path=slice_path)

    # Write summary to file
    log_file_path = os.path.join(plot_dir, "eval_summary.txt")
    with open(log_file_path, "w") as f:
        f.write("\n".join(results_log))
    print(f"Saved summary to {log_file_path}")

if __name__ == "__main__":
    main()
