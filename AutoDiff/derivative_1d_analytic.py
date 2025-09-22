import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.func import vmap, jacfwd, jacrev
from model import CoordinateNet_ordinary as CoordinateNet
from utilities import ackley_1d, gaussian_mixture_1d, mixture_hyperrectangles

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


def evaluate_all():
    root_model_dir = "../models/1d"
    save_dir = "plots_eval_analytic"
    os.makedirs(save_dir, exist_ok=True)

    functions = ["ackley", "gm", "hr"]
    orders = [1, 2, 3]
    N = 2048
    x_vals = torch.linspace(-1, 1, N).view(-1, 1).cuda()
    x_np = x_vals.cpu().numpy()

    for func in functions:
        for order in orders:
            print(f"\nEvaluating {func} | Order {order}")
            model_path = os.path.join(root_model_dir, f"{func}_order={order}.pth")
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}")
                continue

            model = load_model(model_path)
            pred = chunked_derivative(model, x_vals, order).reshape(-1, 1)
            gt = get_ground_truth(func, x_np).reshape(-1, 1)

            mse = np.mean((pred - gt) ** 2)
            print(f"MSE: {mse:.9f}")

            plt.figure(figsize=(10, 4))
            plt.plot(gt, label="Ground Truth", linewidth=1)
            plt.plot(pred, '--', label=f"Predicted (Order {order})", linewidth=1)
            plt.title(f"{func.upper()} | Order {order} Derivative")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(False)
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{func}_order{order}.png")
            plt.savefig(save_path)
            plt.close()

if __name__ == "__main__":
    evaluate_all()
