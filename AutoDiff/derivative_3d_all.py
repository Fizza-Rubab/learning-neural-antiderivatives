import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities import mesh_to_sdf_tensor_eval, save_mesh
from model import CoordinateNet_ordinary as CoordinateNet
from torch.func import vmap, jacfwd, jacrev
import time

def pad_sdf(sdf_volume, pad_fraction=0.3, constant_value=1.0):
    d, h, w = sdf_volume.shape[:3]
    pd, ph, pw = int(d * pad_fraction), int(h * pad_fraction), int(w * pad_fraction)
    padding = ((pd, pd), (ph, ph), (pw, pw))
    if sdf_volume.ndim == 4:
        padding += ((0, 0),)
    return np.pad(sdf_volume, padding, mode='constant', constant_values=constant_value)


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

def evaluate_model_sdf(net_path, mesh_path, order, size=256, chunk_size=4096):
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights['pe'], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    gt = mesh_to_sdf_tensor_eval(mesh_path, size).astype(np.float32)
    print("gt.shape", gt.shape)
    gt = pad_sdf(gt)
    print("gt.shape", gt.shape)
    gt = gt[::4, ::4, ::4]
    print("gt.shape", gt.shape)
    gt = torch.from_numpy(gt).cuda()
    D, H, W = gt.shape

    coords = build_3d_grid(D, H, W)
    pred = []
    x = 1
    for i in range(0, coords.shape[0], chunk_size):
        print(x)
        chunk = coords[i:i+chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        pred.append(out.detach().cpu())
        x+=1
    pred = torch.cat(pred, dim=0).view(D, H, W)
    print("pred.shape", pred.shape)

    mse = torch.mean((pred - gt.cpu()) ** 2).item()
    return pred.numpy(), gt.cpu().numpy(), mse


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


def main():
    mesh_dir = "../data/geometry"
    ckpt_root = "../models/AutoDiff/3d"
    eval_dir = "evaluation_3d_updated"
    plot_dir = os.path.join(eval_dir, "plots")
    mesh_out_dir = os.path.join(eval_dir, "meshes")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(mesh_out_dir, exist_ok=True)

    log_lines = []

    for mesh_file in os.listdir(mesh_dir):
        if not mesh_file.endswith(".ply"):
            continue

        base_name = os.path.splitext(mesh_file)[0]
        mesh_path = os.path.join(mesh_dir, mesh_file)

        for order in [1, 2]:
            print(f"File: {base_name}, Order: {order}", flush=True)
            st = time.time()
            ckpt_path = os.path.join(ckpt_root, f"{base_name}_order={order}.pth")
            if not os.path.isfile(ckpt_path):
                print(f"Skipping missing checkpoint: {ckpt_path}")
                continue

            try:
                pred, gt, mse = evaluate_model_sdf(ckpt_path, mesh_path, order, chunk_size=4096)
            except Exception as e:
                print(f"Error on {base_name} order {order}: {e}")
                continue

            slice_path = os.path.join(plot_dir, f"{base_name}_order{order}.png")
            plot_sdf_slice(pred, gt, z_idx=pred.shape[0] // 2, save_name=slice_path)

            os.makedirs(os.path.join(mesh_out_dir, f"{base_name}_order{order}"), exist_ok=True)
            try:
                save_mesh(pred, os.path.join(mesh_out_dir, f"{base_name}_order{order}"))
            except Exception as e:
                print("Mesh couldn't be saved:", e)

            line = f"{base_name}, order={order}, MSE={mse:.6f}\n"
            print(line.strip())
            log_lines.append(line)
            print(time.time() - st, "elapsed", flush=True)

    mse_log_path = os.path.join(eval_dir, "mse_results.txt")
    with open(mse_log_path, 'a') as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    main()
