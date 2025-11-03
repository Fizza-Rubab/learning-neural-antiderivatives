import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.func import vmap, jacfwd, jacrev
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips

from model import CoordinateNet_ordinary as CoordinateNet
from utilities import ackley_3d, gaussian_mixture_3d, mixture_hyperrectangles, save_mesh


def build_3d_grid(D, H, W):
    zs = torch.linspace(-1, 1, D)
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack([x, y, z], dim=-1)
    return grid.view(-1, 3).cuda()


def nth_derivative(model, x, order):
    """Compute nth derivative for 3D analytic functions."""
    if order == 1:
        return vmap(jacfwd(jacrev(jacfwd(lambda a, b, c: model(torch.cat([a, b, c], -1)),
                                         argnums=0), argnums=1), argnums=2))(
            x[:, 0:1], x[:, 1:2], x[:, 2:3]
        ).reshape(-1, 1)
    elif order == 2:
        return vmap(jacfwd(jacrev(jacfwd(jacfwd(jacrev(jacfwd(
            lambda a, b, c: model(torch.cat([a, b, c], -1)), argnums=0),
            argnums=1), argnums=2), argnums=0), argnums=1), argnums=2))(
            x[:, 0:1], x[:, 1:2], x[:, 2:3]
        ).reshape(-1, 1)
    else:
        raise ValueError("Only 1st or 2nd order supported.")


def chunked_derivative(model, coords, order, chunk_size=8192):
    outputs = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def load_model(path):
    weights = torch.load(path)
    model = CoordinateNet(
        weights["output"], weights["activation"], weights["input"],
        weights["channels"], weights["layers"], weights["encodings"],
        weights["normalize_pe"], weights["pe"], norm_exp=0
    ).cuda()
    model.load_state_dict(weights["ckpt"])
    model.eval()
    return model


def get_ground_truth(func, coords):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    if func == "ackley":
        return ackley_3d(x, y, z)
    elif func == "gm":
        return gaussian_mixture_3d(seed=100)(coords)
    elif func == "hr":
        return mixture_hyperrectangles(coords, dim=3, seed=100, num_rects=45)
    else:
        raise ValueError(f"Unknown function: {func}")


def plot_sdf_slice(pred, gt, save_path):
    """Visualize a central slice of the 3D volume."""
    z_idx = pred.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pred[z_idx], cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title("Predicted")
    axes[1].imshow(gt[z_idx], cmap='seismic', vmin=-1, vmax=1)
    axes[1].set_title("Ground Truth")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_analytic_3d():
    BASE_ROOT = r"D:\learning-neural-antiderivatives\models"
    MODE = "DoG-Noblur"  # or "DoG-Noblur"
    model_dir = os.path.join(BASE_ROOT, MODE, "3d")
    eval_dir = f"evaluation_analytic_3d_{MODE.lower()}"
    os.makedirs(eval_dir, exist_ok=True)

    loss_fn = lpips.LPIPS(net='alex').cuda()

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    print(f"Found {len(model_files)} models in {model_dir}")

    total_mse, count = 0, 0
    logs = []

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)

        # --- Extract function name ---
        func_match = re.match(r"(ackley|gm|hr)", model_file.lower())
        if not func_match:
            print(f"Skipping {model_file} (no func match)")
            continue
        func = func_match.group(1)

        # --- Extract last order ---
        order_matches = re.findall(r"order[_=]?(\d+)", model_file)
        order = int(order_matches[-1]) if order_matches else 0

        # --- Extract scale ---
        scale_match = re.search(r"scale[_=]?([0-9.]+)", model_file)
        if scale_match:
            scale_str = scale_match.group(1).rstrip("._")
            try:
                scale = float(scale_str)
            except ValueError:
                print(f"Bad scale '{scale_str}', defaulting to 0.2")
                scale = 0.2
        else:
            alt_match = re.search(r"[_=]([0-9.]+)\.pth", model_file)
            scale = float(alt_match.group(1)) if alt_match else 0.2

        print(f"\nEvaluating {model_file} | func={func}, order={order}, scale={scale}")

        try:
            model = load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        size = 128
        coords = build_3d_grid(size, size, size)
        coords_np = coords.detach().cpu().numpy()
        gt = get_ground_truth(func, coords_np).reshape(size, size, size)

        try:
            pred = chunked_derivative(model, coords, order + 1).reshape(size, size, size)
        except Exception as e:
            print(f"Derivative computation failed: {e}")
            continue

        if order == 0:
            pred = pred / -scale ** 3
        elif order == 1:
            pred = pred / ((scale ** 3) ** 2)
        elif order == 2:
            pred = pred / -((scale ** 3) ** 3)

        mse = np.mean((pred - gt) ** 2)
        total_mse += mse
        count += 1
        print(f"MSE: {mse:.8f}")

        # Save slice
        slice_path = os.path.join(eval_dir, f"{func}_order{order}_scale{scale}.png")
        plot_sdf_slice(pred, gt, slice_path)

        try:
            save_mesh(pred, os.path.join(eval_dir, f"{func}_order{order}_scale{scale}_mesh"))
        except Exception as e:
            print(f"Mesh save failed: {e}")

        logs.append(f"{model_file}, func={func}, order={order}, scale={scale}, MSE={mse:.8e}\n")

    if count > 0:
        avg_mse = total_mse / count
        with open(os.path.join(eval_dir, "results_summary.txt"), "w") as f:
            f.writelines(logs)
        print(f"Done — {count} models evaluated.")
        print(f"Average MSE = {avg_mse:.8f}")
    else:
        print("No valid models were evaluated.")


if __name__ == "__main__":
    evaluate_analytic_3d()
