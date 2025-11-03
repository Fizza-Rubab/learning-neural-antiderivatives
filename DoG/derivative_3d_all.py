import os
import re
import glob
import torch
import numpy as np
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from utilities import mesh_to_sdf_tensor, save_mesh
from model import CoordinateNet_ordinary as CoordinateNet
from torch.func import vmap, jacfwd, jacrev


def pad_sdf(sdf_volume, pad_fraction=0.3, constant_value=1.0):
    """Pad SDF volume symmetrically with constant value."""
    d, h, w = sdf_volume.shape[:3]
    pd, ph, pw = int(d * pad_fraction), int(h * pad_fraction), int(w * pad_fraction)
    padding = ((pd, pd), (ph, ph), (pw, pw))
    if sdf_volume.ndim == 4:
        padding += ((0, 0),)
    return np.pad(sdf_volume, padding, mode='constant', constant_values=constant_value)


def build_3d_grid(D, H, W):
    """Create a normalized 3D grid of coordinates in [-1, 1]^3."""
    zs = torch.linspace(-1, 1, D)
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack([x, y, z], dim=-1)
    return grid.view(-1, 3).cuda()


def nth_derivative(model, x, order):
    """Compute nth derivative for 3D function using functorch."""
    if order == 0:
        return vmap(jacfwd(jacrev(jacfwd(lambda a, b, c:
               model(torch.cat([a, b, c], -1)), argnums=0),
               argnums=1), argnums=2))(x[:, 0:1], x[:, 1:2], x[:, 2:3]).reshape(-1, 1)
    elif order == 1:
        return vmap(jacfwd(jacrev(jacfwd(jacfwd(jacrev(jacfwd(lambda a, b, c:
               model(torch.cat([a, b, c], -1)), argnums=0), argnums=1),
               argnums=2), argnums=0), argnums=1), argnums=2))(
               x[:, 0:1], x[:, 1:2], x[:, 2:3]).reshape(-1, 1)
    else:
        raise ValueError("Only up to 2nd derivative supported for DoG models.")


def plot_sdf_slice(pred, gt, z_idx, save_name):
    """Plot a middle-plane slice comparison between predicted and GT SDF."""
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


def evaluate_model_sdf(net_path, mesh_path, order, scale=0.1, size=256, chunk_size=4096):
    """Evaluate a 3D model against a ground truth mesh SDF."""
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights['pe'], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    gt = mesh_to_sdf_tensor(mesh_path, size).astype(np.float32)
    gt = pad_sdf(gt)
    gt = gt[::4, ::4, ::4]
    gt_tensor = torch.from_numpy(gt).cuda()

    D, H, W = gt_tensor.shape
    coords = build_3d_grid(D, H, W)

    preds = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        preds.append(out.detach().cpu())
    pred = torch.cat(preds, dim=0).view(D, H, W)

    if order == 0:
        pred = pred / -(scale ** 3)
    elif order == 1:
        pred = pred / ((scale ** 3) ** 2)
    elif order == 2:
        pred = pred / -((scale ** 3) ** 3)

    mse = torch.mean((pred - gt_tensor.cpu()) ** 2).item()
    return pred.numpy(), gt_tensor.cpu().numpy(), mse


# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------

def main():
    BASE_ROOT = r"../models"
    MODE = "DoG-Blur"   # or "DoG-Noblur"
    ckpt_root = os.path.join(BASE_ROOT, MODE, "3d")
    gt_folder = r"..\data\geometry"
    eval_dir = f"evaluation_3d_{MODE.lower()}"

    plot_dir = os.path.join(eval_dir, "plots")
    mesh_dir = os.path.join(eval_dir, "meshes")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    ckpt_files = glob.glob(os.path.join(ckpt_root, "*.pth"))
    print(f"Found {len(ckpt_files)} checkpoints under {ckpt_root}")

    total_mse = 0.0
    count = 0
    logs = []

    for ckpt_path in ckpt_files:
        model_name = os.path.basename(ckpt_path)

        match_id = re.search(r"([A-Za-z0-9]+_[A-Za-z0-9]+)", model_name)
        base_name = match_id.group(1) if match_id else os.path.splitext(model_name)[0]

        order_matches = re.findall(r"order[_=]?([0-9]+)", model_name)
        order = int(order_matches[-1]) if order_matches else 0

        scale_match = re.search(r"scale[_=]?([0-9.]+)", model_name)
        scale = float(scale_match.group(1)) if scale_match else 0.1

        gt_matches = glob.glob(os.path.join(gt_folder, f"*{base_name}*.ply"))
        if not gt_matches:
            print(f"Missing ground truth mesh for {base_name}")
            continue
        mesh_path = gt_matches[0]

        try:
            pred, gt, mse = evaluate_model_sdf(ckpt_path, mesh_path, order, scale=scale)
        except Exception as e:
            print(f"Error evaluating {base_name}: {e}")
            continue

        total_mse += mse
        count += 1

        log = f"{base_name}, order={order}, scale={scale}, MSE={mse:.6f}"
        print(log)
        logs.append(log + "\n")

        slice_path = os.path.join(plot_dir, f"{base_name}_order{order}_scale{scale}.png")
        plot_sdf_slice(pred, gt, z_idx=pred.shape[0] // 2, save_name=slice_path)

        mesh_out_folder = os.path.join(mesh_dir, f"{base_name}_order{order}_scale{scale}")
        os.makedirs(mesh_out_folder, exist_ok=True)
        try:
            save_mesh(pred, mesh_out_folder)
        except Exception as e:
            print("Warning: Mesh could not be saved:", e)

    summary_path = os.path.join(eval_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(logs)

    avg_mse = total_mse / max(1, count)
    print(f"Evaluation complete — {count} models processed.")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
