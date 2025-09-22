import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utilities import load_mp3  # assuming still needed in env
from model import CoordinateNet_ordinary as CoordinateNet
from torch.func import vmap, jacfwd, jacrev
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips


def pad_image(image, pad_fraction=0.3):
    height, width = image.shape[:2]
    pad_height = int(height * pad_fraction)
    pad_width = int(width * pad_fraction)
    image = np.pad(image, ((0, 0), (pad_width, pad_width), (0, 0)), mode='reflect')
    return np.pad(image, ((pad_height, pad_height), (0, 0), (0, 0)), mode='reflect')


def chunked_derivative(model, coords, order, chunk_size=4096):
    outputs = []
    # x = 1
    for i in range(0, coords.shape[0], chunk_size):
        # print(x)
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
        # x+=1
    return torch.cat(outputs, dim=0).numpy()


def nth_derivative(model, x, order):
    if order == 0:
        return vmap(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1))(
            x[:, 0:1], x[:, 1:2]
        ).reshape(-1, 3)
    elif order == 1:
        return vmap(jacrev(jacfwd(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1), argnums=0), argnums=1))(
            x[:, 0:1], x[:, 1:2]
        ).reshape(-1, 3)
    else:
        raise ValueError("Order 2 is intentionally skipped.")


def build_2d_grid(H, W):
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.stack([grid_y, grid_x], dim=-1)
    return coords.view(-1, 2).cuda()


def evaluate_model(net_path, image_path, order, pad_fraction=0.3):
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights['pe'], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    gt = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
    gt = gt.astype(np.float32) / 255.0
    gt = pad_image(gt, pad_fraction=pad_fraction)
    gt = gt[::2, ::2]
    H, W, _ = gt.shape
    coords = build_2d_grid(H, W)
    pred = chunked_derivative(model, coords, order, chunk_size=4096).reshape(*gt.shape)

    pred_clipped = np.clip(pred, 0, 1)
    gt_clipped = np.clip(gt, 0, 1)

    mse = np.mean((pred - gt) ** 2)
    psnr = compare_psnr(gt_clipped, pred_clipped, data_range=1.0)
    ssim = compare_ssim(gt_clipped, pred_clipped, channel_axis=-1, data_range=1.0)

    loss_fn = lpips.LPIPS(net='alex').cuda()
    pred_lp = torch.tensor(pred_clipped).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
    gt_lp = torch.tensor(gt_clipped).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
    lpips_score = loss_fn(pred_lp, gt_lp).item()

    return pred_clipped, gt_clipped, mse, psnr, ssim, lpips_score


def main():
    image_dir = "../data/images"
    ckpt_root = "../models/FD-Blur/2d"
    eval_dir = "evaluation_2d"
    plot_dir = os.path.join(eval_dir, "results")
    os.makedirs(plot_dir, exist_ok=True)

    all_logs = []

    for fname in os.listdir(image_dir):
        if not fname.endswith(".png"):
            continue

        base_name = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)

        for order in [0, 1]:
            ckpt_path = os.path.join(ckpt_root, f"{base_name}_2d_order_{order}_minimal_0.04_samples_200000_order={order}.pth")
            if not os.path.isfile(ckpt_path):
                print(f"Skipping missing: {ckpt_path}")
                continue

            try:
                pred, gt, mse, psnr, ssim, lpips_val = evaluate_model(ckpt_path, image_path, order)
            except Exception as e:
                print(f"Error evaluating {fname} order {order}: {e}")
                continue

            print("pred.shape", pred.shape)
            # Assume pred.shape = (819, 819, 3)
            crop_size = 512
            pad = 153  # 307 from padding, halved due to ::2 subsampling
            pred_cropped = pred[pad:pad + crop_size, pad:pad + crop_size]
            print("pred.shape cropped", pred_cropped.shape)

            from PIL import Image
            pred_uint8 = (pred_cropped * 255).astype(np.uint8)
            Image.fromarray(pred_uint8).save(os.path.join(plot_dir, f"{base_name}_order{order}.png"))

            # Save visualization
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(pred)
            axes[0].set_title(f"Predicted (order={order})")
            axes[1].imshow(gt)
            axes[1].set_title("Ground Truth")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{base_name}_order{order}.png"))
            plt.close()

            log_line = f"{base_name}, order={order}, MSE={mse:.8f}, PSNR={psnr:.8f}, SSIM={ssim:.8f}, LPIPS={lpips_val:.8f}"
            print(log_line)
            all_logs.append(log_line + "\n")

    mse_log_path = os.path.join(eval_dir, "mse_results.txt")
    with open(mse_log_path, 'a') as f:
        f.writelines(all_logs)


if __name__ == "__main__":
    main()
