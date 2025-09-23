import torch
from torch.func import vmap, jacfwd, jacrev
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utilities import ackley_3d, gaussian_mixture_3d, mixture_hyperrectangles
from model import CoordinateNet_ordinary as CoordinateNet
import lpips
import time
import os
from utilities import mesh_to_sdf_tensor, save_mesh



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


def chunked_derivative(model, coords, order, chunk_size=10000, dim=1):
    outputs = []
    x = 1
    for i in range(0, coords.shape[0], chunk_size):
        print(x)
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
        x+=1
    return torch.cat(outputs, dim=0).numpy()


def get_ground_truth(name, coords):
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    if name == "ackley":
        return ackley_3d(x, y, z)
    elif name == "gm":
        return gaussian_mixture_3d(seed=100)(coords)
    elif name == "hr":
        return mixture_hyperrectangles(coords, dim=3, seed=100, num_rects=45)
    else:
        raise ValueError("Unknown analytic function")



def evaluate_model(net_path, gt_path, func_name, order, size=128, blur=False):
    current_ckpt_path, current_gt_path, func_name, order, blur
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights["pe"], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    coords = build_3d_grid(size, size, size).requires_grad_(True)
    x_np = coords.detach().cpu().numpy()



    if blur:
        gt = np.load(gt_path, allow_pickle=True).item()['res']
        gt = gt[::4, ::4, ::4]


    else:
        # gt = get_ground_truth(func_name, x_np).reshape(size, size, size)
        if func_name == "gm":
            gt = GaussianMixture(gt_path).eval(x_np).reshape(size, size)
        elif func_name == "hr":
            gt = HyperrectangleMixture(gt_path).eval(x_np).reshape(size, size)
        else:
            gt = get_ground_truth(func_name, x_np).reshape(size, size)


    pred = chunked_derivative(model, coords, order+1, chunk_size=4096, dim=1).reshape(size, size, size)

    scale = 0.2
    if order == 0:
        pred = pred / -scale ** 3
    elif order == 1:
        pred = pred / ((scale ** 3) ** 2)
    elif order == 2:
        pred = pred / -((scale ** 3) ** 3)

    mse = np.mean((pred - gt) ** 2)

    return pred, gt, mse


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

if __name__ == "__main__":

    import os

    # save_path = '/HPS/n_ntumba/work/Fizza_project/Experiment2d_analytic_without_blur/visuals'
    save_path = '/HPS/n_ntumba/work/Fizza_project/Experiment3d_analytic_no_blur/visuals'

    os.makedirs(save_path, exist_ok=True)
    #net_path1 = "/HPS/n_ntumba/work/Fizza_project/Experiment3d_analytic_no_blur/ckpt/DoG-Noblur_hr_order=0_0.2/current.pth"
    #net_path2 = '/HPS/n_ntumba/work/Fizza_project/Experiment3d_analytic_no_blur/ckpt/DoG-Noblur_gm_order=0_0.2/current.pth'
    #net_path3 = '/HPS/n_ntumba/work/Fizza_project/Experiment3d_analytic_no_blur/ckpt/DoG-Noblur_ackley_order=0_0.2/current.pth'

    net_path1 = "/HPS/n_ntumba/work/Fizza_project/Experiment3d_analytic_with_blur/DoG_blur_hr_3d_order2_0.6_samples30000_3d_scale_0.2_order_0_gpu/current.pth"
    net_path2 = '/HPS/n_ntumba/work/Fizza_project/Experiment3d_analytic_with_blur/DoG_blur_gm_3d_order2_0.6_samples30000_3d_scale_0.2_order_0_gpu/current.pth'
    net_path3 = '/HPS/n_ntumba/work/Fizza_project/Experiment3d_analytic_with_blur/DoG_blur_ackley_3d_order2_0.6_samples30000_3d_scale_0.2_order_0_gpu/current.pth'

    nets = [net_path1, net_path2, net_path3]
    funcs = ['hr', 'gm', 'ackley']

    #gt_param_path1 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/hr_3d_params.npz'
    ##gt_param_path2 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/gm_3d_params.npz'
    #gt_param_path3 = ''

    gt_param_path1 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_3d=2/hr_3d_order2_0.6_samples30000.npy'
    gt_param_path2 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_3d=2/gm_3d_order2_0.6_samples30000.npy'
    gt_param_path3 = '/HPS/n_ntumba/work/Fizza_project/data/analytic_3d=2/ackley_3d_order2_0.6_samples30000.npy'

    nets = [net_path1, net_path2, net_path3]
    funcs = ['hr', 'gm', 'ackley']
    gt_param = [gt_param_path1, gt_param_path2, gt_param_path3]

    total_mse = 0
    blur = True
    count = 0

    for i in range(len(nets)):
        count += 1

        current_ckpt_path = nets[i]
        current_gt_path = gt_param[i]

        func_name = funcs[i]
        order = 0

        eval_dir = "evaluation_3d"
        plot_dir = os.path.join(eval_dir, "plots")
        mesh_out_dir = os.path.join(eval_dir, "meshes")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(mesh_out_dir, exist_ok=True)


        pred, gt, mse = evaluate_model(current_ckpt_path, current_gt_path, func_name, order, blur=blur)
        print(f"MSE {funcs[i]}:   {mse:.8f}")

        slice_path = os.path.join(plot_dir, f"{func_name}_order{order}.png")
        plot_sdf_slice(pred, gt, z_idx=pred.shape[0] // 2, save_name=slice_path)

        os.makedirs(os.path.join(mesh_out_dir, f"{func_name}_order{order}"), exist_ok=True)
        try:
            save_mesh(pred, os.path.join(mesh_out_dir, f"{func_name}_order{order}"))
        except Exception as e:
            print("Mesh couldn't be saved:", e)




#MSE hr:   31543010.00000000
#MSE gm:   40517232.00000000
#MSE ackley:   4017143296.00000000

