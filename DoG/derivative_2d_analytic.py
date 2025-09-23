import torch
from torch.func import vmap, jacfwd, jacrev
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utilities import ackley_2d, gaussian_mixture_2d, mixture_hyperrectangles
from model import CoordinateNet_ordinary as CoordinateNet
import lpips


from scipy.stats import multivariate_normal
from scipy.stats import norm


class GaussianMixture:
    def __init__(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.means = data['means']
        self.weights = data['weights']
        self.covs = data['covs']
        self.dim = self.means.shape[1] if self.means.ndim > 1 else 1
        self.threshold = data['threshold'] if 'threshold' in data else None
        self.discontinuous = bool(data['discontinuous']) if 'discontinuous' in data else False

    def eval(self, x):
        x = np.atleast_2d(x)
        result = np.zeros(x.shape[0])
        if self.dim == 1:
            for w, m, s in zip(self.weights, self.means, self.covs):
                result += w * norm.pdf(x.ravel(), m, s)
        else:
            for w, m, c in zip(self.weights, self.means, self.covs):
                result += w * multivariate_normal.pdf(x, m, c)
        if self.discontinuous and self.threshold is not None:
            if self.dim == 1:
                mask = (x.ravel() >= self.threshold)
            else:
                mask = np.all(x >= self.threshold, axis=1)
            result *= mask.astype(float)
        return result

class HyperrectangleMixture:
    def __init__(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.centers = data['centers']
        self.sizes = data['sizes']
        self.weights = data['weights']
        self.dim = self.centers.shape[1] if self.centers.ndim > 1 else 1
        self.rotation = bool(data['rotation']) if 'rotation' in data else False
        self.angles = data['angles'] if 'angles' in data else None
        self.rots = data['rotations'] if 'rotations' in data else None

    def eval(self, x):
        x = np.atleast_2d(x)
        result = np.zeros(x.shape[0])
        for i in range(len(self.weights)):
            center = self.centers[i]
            size = self.sizes[i]
            weight = self.weights[i]
            if self.dim == 1:
                inside = np.abs(x[:, 0] - center) <= size / 2
            elif self.dim == 2:
                rel = x - center
                if self.rotation and self.angles is not None:
                    angle = self.angles[i]
                    c, s = np.cos(-angle), np.sin(-angle)
                    rot = np.array([[c, -s], [s, c]])
                    rel = rel @ rot.T
                inside = np.all(np.abs(rel) <= size / 2, axis=1)
            elif self.dim == 3:
                rel = x - center
                if self.rotation and self.rots is not None:
                    rot = self.rots[i]
                    rel = rel @ rot.T
                inside = np.all(np.abs(rel) <= size / 2, axis=1)
            else:
                raise ValueError("Unsupported dimension")
            result += weight * inside.astype(float)
        return result

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
        return vmap(jacrev(jacfwd(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1), argnums=0), argnums=1))(
            x[:, 0:1], x[:, 1:2]
        ).reshape(-1, dim)
    elif order == 3:
        return vmap(jacrev(jacfwd(jacrev(jacfwd(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1), argnums=0), argnums=1), argnums=0), argnums=1))(
            x[:, 0:1], x[:, 1:2]
        ).reshape(-1, dim)
    else:
        raise ValueError("Only orders 1, 2, 3 supported")


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



def evaluate_model(net_path, func_name, order, size=1024, gt_path='', blur=False):

    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights["pe"], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    coords = build_2d_grid(size, size).requires_grad_(True)
    x_np = coords.detach().cpu().numpy()

    if blur:
        gt = np.load(gt_path, allow_pickle=True).item()['res'][..., 0]
    else:

        if func_name == "gm":
            gt = GaussianMixture(gt_path).eval(x_np).reshape(size, size)
        elif func_name == "hr":
            gt = HyperrectangleMixture(gt_path).eval(x_np).reshape(size, size)
        else:
            gt = get_ground_truth(func_name, x_np).reshape(size, size)

    pred = chunked_derivative(model, coords, order, chunk_size=4096, dim=1).reshape(size, size)


    scale = 0.01
    if order - 1 == 0:
        print(f'factor : {(float(scale) ** 2)}')
        # pred = pred / (float(scale) ** 2)
        pred = pred * (float(scale) ** 2)
    elif order - 1 == 1:
        # pred = pred / (((float(scale) ** 2) ** 2))
        pred = pred * (((float(scale) ** 2) ** 2))
    elif order - 1 == 2:
        # pred = pred / (((float(scale) ** 2) ** 3))
        pred = pred * (((float(scale) ** 2) ** 3))


    mse = np.mean((pred - gt) ** 2)

    gt = np.clip(gt, 0., 1.)
    pred = np.clip(pred, 0., 1.)
    psnr = compare_psnr(gt, pred, data_range=1.0)
    ssim = compare_ssim(gt, pred, data_range=1.0)

    loss_fn = lpips.LPIPS(net='alex').cuda()
    pred_lp = torch.tensor(pred).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float().cuda() * 2 - 1
    gt_lp = torch.tensor(gt).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float().cuda() * 2 - 1
    lpips_score = loss_fn(pred_lp, gt_lp).item()

    #print(f"MSE:   {mse:.9f}")
    #print(f"PSNR:  {psnr:.9f}")
    #print(f"SSIM:  {ssim:.9f}")
    #print(f"LPIPS: {lpips_score:.9f}")

    return pred, gt, mse, psnr, ssim, lpips_score


if __name__ == "__main__":
    import os
    # save_path = '/HPS/n_ntumba/work/Fizza_project/Experiment2d_analytic_without_blur/visuals'
    save_path = '/HPS/n_ntumba/work/Fizza_project/Experiment2d_analytic_with_blur/visuals'
    os.makedirs(save_path, exist_ok=True)
    net_path1 = "/HPS/n_ntumba/work/Fizza_project/Experiment2d_analytic_without_blur/ckpts/DoG-Noblur_hr_order=0_0.01/current.pth"
    net_path2 = '/HPS/n_ntumba/work/Fizza_project/Experiment2d_analytic_without_blur/ckpts/DoG-Noblur_gm_order=0_0.01/current.pth'
    net_path3 = '/HPS/n_ntumba/work/Fizza_project/Experiment2d_analytic_without_blur/ckpts/DoG-Noblur_ackley_order=0_0.01/current.pth'

    #net_path1 = "/HPS/n_ntumba/work/Fizza_project/Experiment2d_analytic_with_blur/1st_order/DoG_blur_hr_2d_order2_0.03_samples10000_2d_scale_0.01_order_0/current.pth"
    #net_path2 = '/HPS/n_ntumba/work/Fizza_project/Experiment2d_analytic_with_blur/1st_order/DoG_blur_gm_2d_order2_0.03_samples10000_2d_scale_0.01_order_0/current.pth'
    #net_path3 = '/HPS/n_ntumba/work/Fizza_project/Experiment2d_analytic_with_blur/1st_order/DoG_blur_ackley_2d_order2_0.03_samples10000_2d_scale_0.01_order_0/current.pth'

    nets = [net_path1, net_path2, net_path3]
    funcs = ['hr', 'gm', 'ackley']

    gt_param_path1 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/hr_2d_params.npz'
    gt_param_path2 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/gm_2d_params.npz'
    gt_param_path3 = ''

    #gt_param_path1 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_2d=2/hr_2d_order2_0.03_samples10000.npy'
    #gt_param_path2 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_2d=2/gm_2d_order2_0.03_samples10000.npy'
    #gt_param_path3 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_2d=2/ackley_2d_order2_0.03_samples10000.npy'
    gt_param = [gt_param_path1, gt_param_path2, gt_param_path3]

    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    count = 0
    blur = False

    for i in range(len(nets)):
        count += 1

        current_ckpt_path = nets[i]
        current_gt_path = gt_param[i]

        func_name = funcs[i]
        order = 1
        pred, gt, mse, psnr, ssim, lpips_val = evaluate_model(current_ckpt_path, func_name, order, size=1024, gt_path=current_gt_path, blur=blur)

        total_mse += mse
        total_psnr += psnr
        total_ssim += ssim
        total_lpips += lpips_val

        print(f'function: {funcs[i]}, order: {order}, mse: {mse:.6f}, psnr: {psnr:.6f}, ssim: {ssim:.6f}, lpips: {lpips_val:.6f}')


        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pred, cmap='viridis')
        axes[0].set_title("Predicted")
        axes[1].imshow(gt, cmap='viridis')
        axes[1].set_title("Ground Truth")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"ad_analytic_comparison_{func_name}_order{order}.png"))

    print(f"Total mse: {total_mse:.6f}")
    print(f"Total psnr: {total_psnr:.6f}")
    print(f"Total ssim: {total_ssim:.6f}")
    print(f"Total lpips: {total_lpips:.6f}")

    print()
    print(f'count : {count}')
    print(f'average mse: {total_mse / count:.6f}')
    print(f'average psnr: {total_psnr / count:.6f}')
    print(f'average ssim: {total_ssim / count:.6f}')
    print(f'average lpips: {total_lpips / count:.6f}')
