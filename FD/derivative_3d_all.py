import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from utilities import mesh_to_sdf_tensor_eval, save_mesh
from model import CoordinateNet_ordinary as CoordinateNet
from torch.func import vmap, jacfwd, jacrev
import time


def pad_sdf(sdf_volume, pad_fraction=0.3, constant_value=1.0):
    depth, height, width = sdf_volume.shape[:3]
    pad_depth = int(depth * pad_fraction)
    pad_height = int(height * pad_fraction)
    pad_width = int(width * pad_fraction)
    padding = ((pad_depth, pad_depth), (pad_height, pad_height), (pad_width, pad_width))
    return np.pad(sdf_volume, padding, mode='constant', constant_values=constant_value)


def build_3d_grid(D, H, W):
    zs = torch.linspace(-1, 1, D)
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack([x, y, z], dim=-1)
    return grid.view(-1, 3).cuda()


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
    # x = 1
    for i in range(0, coords.shape[0], chunk_size):
        # print(x)
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
        # x +=1
    return torch.cat(outputs, dim=0).numpy()


def evaluate_model_sdf(net_path, mesh_path, order, size=256, chunk_size=4096):
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'],
        weights['activation'],
        weights['input'],
        weights['channels'],
        weights['layers'],
        weights['encodings'],
        weights['normalize_pe'],
        weights["pe"],
        norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    gt = mesh_to_sdf_tensor_eval(mesh_path, size).astype(np.float32)
    gt = pad_sdf(gt)
    gt = gt[::4, ::4, ::4]
    gt = torch.from_numpy(gt).cuda()
    D, H, W = gt.shape

    coords = build_3d_grid(D, H, W)
    pred = chunked_derivative(model, coords, order, chunk_size).reshape(D, H, W)
    if order == 0:
        pred *=-1
    mse = np.mean((pred - gt.cpu().numpy()) ** 2)
    return pred, gt.cpu().numpy(), mse


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
    mesh_dir = "/HPS/antiderivative_project/work/data/geometry"
    ckpt_root = "/HPS/antiderivative_project/work/NFC-Blur/experiments/results_3d"
    eval_dir = "evaluation_3d_updated"
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    mse_log_path = os.path.join(eval_dir, "mse_results.txt")
    with open(mse_log_path, 'a') as f:
        for mesh_file in os.listdir(mesh_dir):
            if not mesh_file.endswith(".ply"):
                continue

            base_name = os.path.splitext(mesh_file)[0]
            mesh_path = os.path.join(mesh_dir, mesh_file)

            for order in [1]:  # Only orders 0 and 1
                print(f"File: {base_name}, Order: {order}", flush=True)
                st = time.time()
                ckpt_path = os.path.join(ckpt_root, f"NFC-Blur_{base_name}_3d_order_{order}_0.04_samples_10000_order={order}", "current.pth")
                if not os.path.exists(ckpt_path):
                    print(f"Skipping missing checkpoint: {ckpt_path}")
                    continue

                try:
                    pred, gt, mse = evaluate_model_sdf(ckpt_path, mesh_path, order, chunk_size=4096)
                except Exception as e:
                    print(f"Error in {base_name} order {order}: {e}")
                    continue

                print(f"{base_name}, order={order}, MSE={mse:.6f}")
                f.write(f"{base_name}, order={order}, MSE={mse:.6f}\n")

                os.makedirs(os.path.join(plot_dir, f"{base_name}_order{order}"), exist_ok=True)
                save_mesh(pred, os.path.join(plot_dir, f"{base_name}_order{order}"))

                slice_path = os.path.join(plot_dir, f"{base_name}_order{order}.png")
                plot_sdf_slice(pred, gt, save_path=slice_path)
                print(time.time() - st, "elapsed", flush=True)
            #     break
            # break



if __name__ == "__main__":
    main()
