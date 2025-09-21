import torch
from torch.func import vmap, jacfwd, jacrev
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utilities import ackley_3d, gaussian_mixture_3d, mixture_hyperrectangles
from model import CoordinateNet_ordinary as CoordinateNet
import lpips
import time
import os
from utilities import mesh_to_sdf_tensor, save_mesh

def build_3d_grid(D, H, W):
    zs = torch.linspace(-1, 1, D)
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack([x, y, z], dim=-1)
    return grid.view(-1, 3).cuda()

def nth_derivative(model, x, order):
    if order == 1:
        derivative = vmap(jacfwd(jacrev(jacfwd(lambda a, b, c: model(torch.cat([a, b, c], -1)), argnums=0), argnums=1), argnums=2))(
                x[:, 0:1], x[:, 1:2], x[:, 2:3]
            ).reshape(-1, 1)
    elif order ==2:
        derivative = vmap(jacfwd(jacrev(jacfwd(jacfwd(jacrev(jacfwd(lambda a, b, c: model(torch.cat([a, b, c], -1)), argnums=0), argnums=1), argnums=2), argnums=0), argnums=1), argnums=2))(
                x[:, 0:1], x[:, 1:2], x[:, 2:3]
            ).reshape(-1, 1)
    elif order ==3:
        derivative = vmap(jacfwd(jacrev(jacfwd(jacfwd(jacrev(jacfwd(jacfwd(jacrev(jacfwd(lambda a, b, c: model(torch.cat([a, b, c], -1)), argnums=0), argnums=1), argnums=2), argnums=0), argnums=1), argnums=2), argnums=0), argnums=1), argnums=2))(
                x[:, 0:1], x[:, 1:2], x[:, 2:3]
            ).reshape(-1, 1)
    return derivative


def chunked_derivative(model, coords, order, chunk_size=10000, dim=1):
    outputs = []
    x = 1
    for i in range(0, coords.shape[0], chunk_size):
        print(x)
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
        x+=1
    return torch.cat(outputs, dim=0).numpy()


def get_ground_truth(name, coords):
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    if name == "ackley":
        return ackley_3d(x, y, z)
    elif name == "gm":
        return gaussian_mixture_3d(seed=100)(coords)
    elif name == "hr":
        return mixture_hyperrectangles(coords, dim=3, seed=100, num_rects=45)
    else:
        raise ValueError("Unknown analytic function")



def evaluate_model(net_path, func_name, order, size=128):
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights["pe"], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    coords = build_3d_grid(size, size, size).requires_grad_(True)
    x_np = coords.detach().cpu().numpy()
    gt = get_ground_truth(func_name, x_np).reshape(size, size, size)

    pred = chunked_derivative(model, coords, order, chunk_size=4096, dim=1).reshape(size, size, size)

    mse = np.mean((pred - gt) ** 2)

    print(f"MSE:   {mse:.8f}")
    return pred, gt


def plot_sdf_slice(pred, gt, z_idx, save_name):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pred[z_idx], cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title("Predicted")
    axes[1].imshow(gt[z_idx], cmap='seismic', vmin=-1, vmax=1)
    axes[1].set_title("Ground Truth")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

if __name__ == "__main__":
    func_name = "hr" 
    net_path = f"/HPS/antiderivative_project/work/Autoint/experiments/results_3d/Autoint_{func_name}_order=2/current.pth"
    order = 2
    eval_dir = "evaluation_3d"
    plot_dir = os.path.join(eval_dir, "plots")
    mesh_out_dir = os.path.join(eval_dir, "meshes")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(mesh_out_dir, exist_ok=True)


    pred, gt = evaluate_model(net_path, func_name, order)
    slice_path = os.path.join(plot_dir, f"{func_name}_order{order}.png")
    plot_sdf_slice(pred, gt, z_idx=pred.shape[0] // 2, save_name=slice_path)

    os.makedirs(os.path.join(mesh_out_dir, f"{func_name}_order{order}"), exist_ok=True)
    try:
        save_mesh(pred, os.path.join(mesh_out_dir, f"{func_name}_order{order}"))
    except Exception as e:
        print("Mesh couldn't be saved:", e)

