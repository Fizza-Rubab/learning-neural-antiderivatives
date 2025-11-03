import os
import re
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from model import CoordinateNet_ordinary as CoordinateNet
from torch.func import vmap, jacfwd, jacrev
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips


def pad_image(image, pad_fraction=0.3):
    """Symmetrically pad an image by a fraction of its size."""
    h, w = image.shape[:2]
    ph, pw = int(h * pad_fraction), int(w * pad_fraction)
    image = np.pad(image, ((0, 0), (pw, pw), (0, 0)), mode='reflect')
    image = np.pad(image, ((ph, ph), (0, 0), (0, 0)), mode='reflect')
    return image


def build_2d_grid(H, W):
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.stack([grid_y, grid_x], dim=-1)
    return coords.view(-1, 2).cuda()


def nth_derivative(model, x, order):
    """Compute nth derivative of the model output wrt 2D inputs."""
    if order == 0:
        return vmap(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)),
                                  argnums=0), argnums=1))(x[:, 0:1], x[:, 1:2]).reshape(-1, 3)
    elif order == 1:
        return vmap(
            jacrev(jacfwd(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)),
                                        argnums=0), argnums=1),
                          argnums=0), argnums=1)
        )(x[:, 0:1], x[:, 1:2]).reshape(-1, 3)
    else:
        raise ValueError("Only orders 1 and 2 are supported for DoG models.")


def chunked_derivative(model, coords, order, chunk_size=4096):
    """Compute derivatives in chunks to avoid GPU OOM."""
    outputs = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def evaluate_model(net_path, image_path, order, lpips_fn):
    """Run model evaluation on a single image."""
    scale_match = re.search(r"scale[_=]?([0-9.]+)", os.path.basename(net_path))
    scale = float(scale_match.group(1)) if scale_match else 0.01

    # Load model weights
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights['pe'], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    gt = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    gt = gt.astype(np.float32) / 255.0
    gt = pad_image(gt)
    gt = gt[::2, ::2]

    H, W, _ = gt.shape
    coords = build_2d_grid(H, W)
    pred = chunked_derivative(model, coords, order).reshape(H, W, 3)

    if order == 0:
        pred *= (scale ** 2)
    elif order == 1:
        pred *= (scale ** 2) ** 2
    elif order == 2:
        pred *= (scale ** 3)  ** 3

    # Clip for visualization
    pred_clip = np.clip(pred, 0, 1)
    gt_clip = np.clip(gt, 0, 1)

    # Metrics
    mse = np.mean((pred_clip - gt_clip) ** 2)
    psnr = compare_psnr(gt_clip, pred_clip, data_range=1.0)
    ssim = compare_ssim(gt_clip, pred_clip, channel_axis=2, data_range=1.0)
    pred_lp = torch.tensor(pred_clip).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
    gt_lp = torch.tensor(gt_clip).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
    lpips_score = lpips_fn(pred_lp, gt_lp).item()

    return pred_clip, gt_clip, mse, psnr, ssim, lpips_score, scale


def main():
    BASE_ROOT = r"../models"
    MODE = "DoG-Noblur"
    ckpt_root = os.path.join(BASE_ROOT, MODE, "2d")
    image_dir = r"..\data\images"
    eval_dir = f"evaluation_2d_{MODE.lower()}"
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    lpips_fn = lpips.LPIPS(net='alex').cuda()
    logs = []

    ckpt_files = glob.glob(os.path.join(ckpt_root, "*.pth"))
    print(f"Found {len(ckpt_files)} checkpoints under {ckpt_root}")

    for ckpt_path in ckpt_files:
        model_name = os.path.basename(ckpt_path)

        img_match = re.search(r"(\d{3,4})", model_name)
        if not img_match:
            print(f"Skipping {model_name} (no image ID found)")
            continue
        img_id = img_match.group(1)

        image_path = os.path.join(image_dir, f"{img_id}.png")
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue

        order = 0

        try:
            pred, gt, mse, psnr, ssim, lpips_val, scale = evaluate_model(ckpt_path, image_path, order, lpips_fn)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

        log = (f"{model_name}, img={img_id}, scale={scale}, order={order}, "
               f"MSE={mse:.6f}, PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips_val:.4f}")
        print(log)
        logs.append(log + "\n")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pred)
        axes[0].set_title(f"Predicted (order={order})")
        axes[1].imshow(gt)
        axes[1].set_title("Ground Truth")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{img_id}_order{order}_scale{scale}.png"))
        plt.close()

    # Save summary
    summary_path = os.path.join(eval_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(logs)

    print(f"Evaluation complete. Results saved to {summary_path}")


if __name__ == "__main__":
    main()
