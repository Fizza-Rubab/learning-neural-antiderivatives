import torch
from torch.func import vmap, jacfwd, jacrev
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utilities import ackley_2d, gaussian_mixture_2d, mixture_hyperrectangles
from model import CoordinateNet_ordinary as CoordinateNet
import lpips
import time


def build_2d_grid(H, W):
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.stack([grid_y, grid_x], dim=-1)
    return coords.view(-1, 2).cuda()


def nth_derivative(model, x, order, dim=1):
    if order == 1:
        return vmap(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1))(
            x[:, 0:1], x[:, 1:2]
        ).reshape(-1, dim)
    elif order == 2:
        def reduced_derivative(a, b):
            out = model(torch.cat([a, b], -1))
            term = (
                a * b * out[:1]
                - a * out[1:2]
                - b * out[2:3]
                + out[3:4]
            )
            return term

        return vmap(jacrev(jacfwd(jacrev(jacfwd(reduced_derivative, argnums=0), argnums=1), argnums=0), argnums=1))(
            x[:, 0:1], x[:, 1:2]
        ).reshape(-1, dim)


def chunked_derivative(model, coords, order, chunk_size=10000, dim=1):
    outputs = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order, dim)
        outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def get_ground_truth(name, coords):
    x = coords[:, 0]
    y = coords[:, 1]
    if name == "ackley":
        return ackley_2d(x, y)
    elif name == "gm":
        return gaussian_mixture_2d(seed=100)(coords)
    elif name == "hr":
        return mixture_hyperrectangles(coords, dim=2, seed=100, num_rects=5, rotation=True)
    else:
        raise ValueError("Unknown analytic function")



def evaluate_model(net_path, func_name, order, size=1024):
    # Load model
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights["pe"], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    # Build domain and GT
    coords = build_2d_grid(size, size).requires_grad_(True)
    x_np = coords.detach().cpu().numpy()
    gt = get_ground_truth(func_name, x_np).reshape(size, size)

    pred = chunked_derivative(model, coords, order, chunk_size=4096, dim=1).reshape(size, size)

    mse = np.mean((pred - gt) ** 2)

    pred = np.clip(pred, 0., 1.)
    gt = np.clip(gt, 0., 1.)

    psnr = compare_psnr(gt, pred, data_range=1.0)
    ssim = compare_ssim(gt, pred, data_range=1.0)

    loss_fn = lpips.LPIPS(net='alex').cuda()
    pred_lp = torch.tensor(pred).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float().cuda() * 2 - 1
    gt_lp = torch.tensor(gt).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float().cuda() * 2 - 1
    lpips_score = loss_fn(pred_lp, gt_lp).item()

    print(pred.min(), pred.max(), gt.min(), gt.max())

    print(f"MSE:   {mse:.9f}")
    print(f"PSNR:  {psnr:.9f}")
    print(f"SSIM:  {ssim:.9f}")
    print(f"LPIPS: {lpips_score:.9f}")

    return pred, gt


if __name__ == "__main__":
    func_name = "hr"  # or "gm", "hr"
    net_path = f"../models/Reduction/2d/{func_name}_order=2.pth"
    order = 2

    pred, gt = evaluate_model(net_path, func_name, order)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pred, cmap='viridis')
    axes[0].set_title("Predicted")
    axes[1].imshow(gt, cmap='viridis')
    axes[1].set_title("Ground Truth")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"ad_analytic_comparison_{func_name}_order{order}.png")
