import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from model import CoordinateNet_ordinary as CoordinateNet
from torch.func import vmap, jacfwd, jacrev
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips


def pad_image(image, pad_fraction=0.3):
    h, w = image.shape[:2]
    ph, pw = int(h * pad_fraction), int(w * pad_fraction)
    image = np.pad(image, ((0, 0), (pw, pw), (0, 0)), mode='reflect')
    image = np.pad(image, ((ph, ph), (0, 0), (0, 0)), mode='reflect')
    return image


def build_2d_grid(H, W):
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
    coords = torch.stack([grid_x, grid_y], dim=-1)  # Use (x, y) order
    return coords.view(-1, 2).cuda()


def nth_derivative(model, x, order):
    if order == 1:
        return vmap(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1))(
                x[:, 0:1], x[:, 1:2]
            ).reshape(-1, 3)
    elif order ==2:
        return vmap(jacrev(jacfwd(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1), argnums=0), argnums=1))(
                x[:, 0:1], x[:, 1:2]
            ).reshape(-1, 3)
    elif order ==3:
        return vmap(jacrev(jacfwd(jacrev(jacfwd(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1), argnums=0), argnums=1), argnums=0), argnums=1))(
                x[:, 0:1], x[:, 1:2]
            ).reshape(-1, 3)
    else:
        raise ValueError("Only orders 1, 2, 3 supported.")


def chunked_derivative(model, coords, order, chunk_size=4096):
    outputs = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def evaluate_model(net_path, image_path, order, lpips_fn):
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

    pred = chunked_derivative(model, coords, order)
    pred = pred.reshape(H, W, 3)

    pred_clip = np.clip(pred, 0, 1)
    gt_clip = np.clip(gt, 0, 1)

    mse = np.mean((pred - gt) ** 2)
    psnr = compare_psnr(gt_clip, pred_clip, data_range=1.0)
    ssim = compare_ssim(gt_clip, pred_clip, channel_axis=2, data_range=1.0)

    pred_lp = torch.tensor(pred_clip).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
    gt_lp = torch.tensor(gt_clip).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
    lpips_score = lpips_fn(pred_lp, gt_lp).item()

    return pred_clip, gt_clip, mse, psnr, ssim, lpips_score


def main():
    image_dir = "/HPS/antiderivative_project/work/data/images"
    ckpt_root = "/HPS/antiderivative_project/work/NFC-MC/experiments/results_2d"
    eval_dir = "evaluation_2d"
    plot_dir = os.path.join(eval_dir, "results")
    os.makedirs(plot_dir, exist_ok=True)

    all_logs = []
    lpips_fn = lpips.LPIPS(net='alex').cuda()

    ims = os.listdir(image_dir)
    for img_file in ims:
        if not img_file.endswith(".png"):
            continue

        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)
        print(f"Image: {base_name}.png")
        for order in [2]:
            ckpt_path = os.path.join(ckpt_root, f"NFC-MC_{base_name}_order={order}", "current.pth")
            if not os.path.exists(ckpt_path):
                print(f"Missing: {ckpt_path}")
                continue

            try:
                pred, gt, mse, psnr, ssim, lpips_val = evaluate_model(ckpt_path, img_path, order, lpips_fn)
            except Exception as e:
                print(f"Error on {base_name} order {order}: {e}")
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


    #         fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    #         axes[0].imshow(pred)
    #         axes[0].set_title(f"Autoint AD (order={order})")
    #         axes[1].imshow(gt)
    #         axes[1].set_title("Ground Truth")
    #         for ax in axes:
    #             ax.axis("off")
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(plot_dir, f"{base_name}_order{order}.png"))
    #         plt.close()

    #         log = (f"{base_name}, order={order}, "
    #                f"MSE={mse:.8f}, PSNR={psnr:.8f}, "
    #                f"SSIM={ssim:.8f}, LPIPS={lpips_val:.8f}")
    #         print(log)
    #         all_logs.append(log + "\n")

    # mse_log_path = os.path.join(eval_dir, "mse_results.txt")
    # with open(mse_log_path, 'a') as f:
    #     f.writelines(all_logs)


if __name__ == "__main__":
    main()