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
from utilities import ackley_2d, gaussian_mixture_2d, mixture_hyperrectangles

def build_2d_grid(H, W):
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.stack([grid_y, grid_x], dim=-1)
    return coords.view(-1, 2).cuda()


def nth_derivative(model, x, order):
    if order == 0:
        return vmap(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)),
                                  argnums=0), argnums=1))(x[:, 0:1], x[:, 1:2]).reshape(-1, 1)
    elif order == 1:
        return vmap(jacrev(jacfwd(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)),
                                                argnums=0), argnums=1), argnums=0), argnums=1))(x[:, 0:1], x[:, 1:2]).reshape(-1, 1)
    else:
        raise ValueError("Only 1st and 2nd order supported")


def chunked_derivative(model, coords, order, chunk_size=4096):
    outputs = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def load_model(model_path):
    weights = torch.load(model_path)
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
    return model


def get_ground_truth(func, coords):
    x = coords[:, 0]
    y = coords[:, 1]
    if func == "ackley":
        return ackley_2d(x, y)
    elif func == "gm":
        return gaussian_mixture_2d(seed=100)(coords)
    elif func == "hr":
        return mixture_hyperrectangles(coords, dim=2, seed=100, num_rects=5, rotation=True)
    else:
        raise ValueError(f"Unknown analytic function: {func}")

def evaluate_analytic_2d():
    BASE_ROOT = r"D:\learning-neural-antiderivatives\models"
    MODE = "DoG-Noblur"  # or "DoG-Noblur"
    model_dir = os.path.join(BASE_ROOT, MODE, "2d")
    eval_dir = f"evaluation_analytic_2d_{MODE.lower()}"
    os.makedirs(eval_dir, exist_ok=True)

    loss_fn = lpips.LPIPS(net='alex').cuda()

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    print(f"Found {len(model_files)} models in {model_dir}")

    total_mse, total_psnr, total_ssim, total_lpips = 0, 0, 0, 0
    count = 0
    logs = []

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)

        func_match = re.match(r"(ackley|gm|hr)", model_file.lower())
        if not func_match:
            print(f"Skipping {model_file} (no valid function name)")
            continue
        func = func_match.group(1)

        order_matches = re.findall(r"order[_=]?(\d+)", model_file)
        order = int(order_matches[-1]) if order_matches else 0

        scale_match = re.search(r"scale[_=]?([0-9.]+)", model_file)
        if scale_match:
            scale_str = scale_match.group(1).rstrip("._")
            try:
                scale = float(scale_str)
            except ValueError:
                print(f"Invalid scale '{scale_str}', defaulting to 0.01")
                scale = 0.01
        else:
            alt_match = re.search(r"[_=]([0-9.]+)\.pth", model_file)
            scale = float(alt_match.group(1)) if alt_match else 0.01

        print(f"\nEvaluating {model_file} | func={func}, order={order}, scale={scale}")

        try:
            model = load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        size = 256
        coords = build_2d_grid(size, size)
        coords_np = coords.detach().cpu().numpy()
        gt = get_ground_truth(func, coords_np).reshape(size, size)

        # Predict
        try:
            pred = chunked_derivative(model, coords, order).reshape(size, size)
        except Exception as e:
            print(f"Derivative failed: {e}")
            continue

        if order == 0:
            pred *= (scale ** 2)
        elif order == 1:
            pred *= (scale ** 2) ** 2
        elif order == 2:
            pred *= (scale ** 2) ** 3

        pred_clip = np.clip(pred, 0, 1)
        gt_clip = np.clip(gt, 0, 1)

        mse = np.mean((pred_clip - gt_clip) ** 2)
        psnr = compare_psnr(gt_clip, pred_clip, data_range=1.0)
        ssim = compare_ssim(gt_clip, pred_clip, data_range=1.0)
        pred_lp = torch.tensor(pred_clip).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float().cuda() * 2 - 1
        gt_lp = torch.tensor(gt_clip).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float().cuda() * 2 - 1
        lpips_val = loss_fn(pred_lp, gt_lp).item()

        total_mse += mse
        total_psnr += psnr
        total_ssim += ssim
        total_lpips += lpips_val
        count += 1

        print(f"MSE={mse:.6e}, PSNR={psnr:.3f}, SSIM={ssim:.3f}, LPIPS={lpips_val:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pred_clip, cmap='viridis')
        axes[0].set_title(f"Predicted ({func})")
        axes[1].imshow(gt_clip, cmap='viridis')
        axes[1].set_title("Ground Truth")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f"{func}_order{order}_scale{scale}.png"))
        plt.close()

        logs.append(f"{model_file}, func={func}, order={order}, scale={scale}, "
                    f"MSE={mse:.8e}, PSNR={psnr:.3f}, SSIM={ssim:.3f}, LPIPS={lpips_val:.3f}\n")

    if count > 0:
        with open(os.path.join(eval_dir, "results_summary.txt"), "w") as f:
            f.writelines(logs)

        print(f"Done — {count} valid models evaluated.")
        print(f"Average MSE={total_mse/count:.6e}, PSNR={total_psnr/count:.3f}, "
              f"SSIM={total_ssim/count:.3f}, LPIPS={total_lpips/count:.3f}")
    else:
        print("No valid models were evaluated.")


if __name__ == "__main__":
    evaluate_analytic_2d()
