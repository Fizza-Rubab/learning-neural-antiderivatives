import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.func import vmap, jacfwd, jacrev
from model import CoordinateNet_ordinary as CoordinateNet
from utilities import ackley_1d, gaussian_mixture_1d, mixture_hyperrectangles
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



def nth_derivative(model, x, order):
    if order == 1:
        return vmap(jacfwd(model))(x)
    elif order == 2:
        return vmap(jacrev(jacfwd(model)))(x)
    elif order == 3:
        return vmap(jacfwd(jacrev(jacfwd(model))))(x)
    else:
        raise ValueError("Only orders 0–2 are supported")

def chunked_derivative(model, coords, order, chunk_size=2048):
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

def get_ground_truth(func_name, x_vals):
    if func_name == "ackley":
        return ackley_1d(x_vals).reshape(-1, 1)
    elif func_name == "gm":
        return gaussian_mixture_1d(seed=100)(x_vals).reshape(-1, 1)
    elif func_name == "hr":
        return mixture_hyperrectangles(x_vals.reshape(-1, 1), dim=1, seed=100).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported function: {func_name}")


def evaluate_all2():
    root_model_dir = "/HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_no_blur/order2/ckpts"
    # save_dir = "/HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_no_blur/order2/visuals2nd"
    save_dir = '/HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_no_blur/order2/visual2nd2/'
    gt_path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/'
    # gt_path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_1d=2/'

    #root_model_dir = "//HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_no_blur/order1/ckpts"
    #save_dir = "/HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_no_blur/order1/visual1st"
    #gt_path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_params/'

    #root_model_dir = "/HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_with_blur/ckpts"
    #save_dir = "/HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_with_blur/visual1st"
    #gt_path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_1d=2/'

    #root_model_dir = "/HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_with_blur/2nd/ckpt"
    #save_dir = "/HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_with_blur/2nd/visual_2nd/"
    #gt_path = '/HPS/n_ntumba/work/Fizza_project/data/analytic_1d_novel=2/'

    blur = False


    gt_dir = os.listdir(gt_path)

    # ackley_1d_order2_0.03_samples100000.npy

    os.makedirs(save_dir, exist_ok=True)
    model_list = os.listdir(root_model_dir)


    total_mse = 0
    count = 0

    for i in range(len(model_list)):
        count += 1

        current_model = model_list[i]
        print(current_model)
        #exit()
        N = 2048
        x_vals = torch.linspace(-1, 1, N).view(-1, 1).cuda()
        x_np = x_vals.cpu().numpy()
        scale = 0.03
        order = 2
        func = current_model.split('_')[1]

        # /HPS/n_ntumba/work/Fizza_project/Experiment1d_analytic_with_blur/2nd/ckpt/DoG-blur_ackley_order=1_scale_0.03/checkpoint_150000.pth
        model_path = os.path.join(root_model_dir, current_model, f'current.pth')
        print(model_path)

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            continue

        model = load_model(model_path)
        pred = chunked_derivative(model, x_vals, order).reshape(-1, 1)#[200:-200]
        gt = get_ground_truth(func, x_np).reshape(-1, 1)# [200:-200]

        if order-1 == 0:
            pred = pred * -scale
        elif order-1 == 1:
            pred = pred * scale ** 2
        elif order-1 == 2:
            pred = pred * -scale ** 3

        if blur:
            gt_obj_path = os.path.join(gt_path, f'{func}_1d_order2_0.09_samples100000.npy')
            gt = np.load(gt_obj_path, allow_pickle=True).item()['res']


        mse = np.mean((pred - gt) ** 2)
        total_mse += mse
        print(f"MSE {func}: {mse:.9f}")

        f = 1
        plt.figure(figsize=(10, 4))
        plt.plot(x_np, gt, label="Ground Truth", linewidth=1)
        plt.plot(x_np[f:-f], pred[f:-f], '--', label=f"Predicted (Order {order})_{scale}", linewidth=1)
        plt.title(f"{func.upper()} | Order {order} Derivative scale {scale}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{func}_order{order}+{scale}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"Total MSE: {total_mse:.20f}")
    print(count)
    print(total_mse / count)


def evaluate_all():
    root_model_dir = "/HPS/n_ntumba/work/Fizza_project/pilot_study/1d_second_order/more_samples/ckpt/"
    save_dir = "/HPS/n_ntumba/work/Fizza_project/pilot_study/1d_second_order/more_samples/plots"
    os.makedirs(save_dir, exist_ok=True)

    models = os.listdir(root_model_dir)

    functions = ["ackley", "gm", "hr"]
    orders = [2]  # [1, 2, 3]
    N = 2048
    x_vals = torch.linspace(-1, 1, N).view(-1, 1).cuda()
    x_np = x_vals.cpu().numpy()
    scale = 0.01

    for func in functions:

        current_model_path = os.path.join(root_model_dir, func)

        print(f"\nEvaluating {func} | Order {order}")
        pattern = f'DoG-Noblur_{func}_order={order - 1}_scale_0.01/current.pth/'
        model_path = os.path.join(root_model_dir, pattern)

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            continue

        model = load_model(model_path)
        pred = -chunked_derivative(model, x_vals, order).reshape(-1, 1)
        gt = get_ground_truth(func, x_np).reshape(-1, 1)

        if order - 1 == 0:
            pred = pred * scale
        elif order - 1 == 1:
            pred = -pred * scale ** 2
        elif order - 1 == 2:
            pred = pred * scale ** 3

        mse = np.mean((pred - gt) ** 2)
        print(f"MSE: {mse:.9f}")

        plt.figure(figsize=(10, 4))
        plt.plot(x_np, gt, label="Ground Truth", linewidth=1)
        plt.plot(x_np, pred, '--', label=f"Predicted (Order {order})", linewidth=1)
        plt.title(f"{func.upper()} | Order {order} Derivative")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{func}_order{order}.png")
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    evaluate_all2()

