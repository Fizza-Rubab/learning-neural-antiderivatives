import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.func import vmap, jacfwd, jacrev

from model import CoordinateNet_ordinary as CoordinateNet
from utilities import ackley_1d, gaussian_mixture_1d, mixture_hyperrectangles

def nth_derivative(model, x, order):
    """Compute nth derivative using functorch."""
    if order == 0:
        return vmap(jacfwd(model))(x)
    elif order == 1:
        return vmap(jacrev(jacfwd(model)))(x)
    elif order == 2:
        return vmap(jacfwd(jacrev(jacfwd(model))))(x)
    else:
        raise ValueError("Only orders 0–2 supported")


def chunked_derivative(model, coords, order, chunk_size=2048):
    """Compute derivatives in chunks to save GPU memory."""
    outputs = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def load_model(model_path):
    """Load a CoordinateNet model from checkpoint."""
    weights = torch.load(model_path)
    model = CoordinateNet(
        weights['output'],
        weights['activation'],
        weights['input'],
        weights['channels'],
        weights['layers'],
        weights['encodings'],
        weights['normalize_pe'],
        weights['pe'],
        norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()
    return model


def get_ground_truth(func_name, x_vals):
    """Compute ground truth analytic function."""
    if func_name == "ackley":
        return ackley_1d(x_vals).reshape(-1, 1)
    elif func_name == "gm":
        return gaussian_mixture_1d(seed=100)(x_vals).reshape(-1, 1)
    elif func_name == "hr":
        return mixture_hyperrectangles(x_vals.reshape(-1, 1), dim=1, seed=100).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported analytic function: {func_name}")



def evaluate_analytic_models():
    BASE_ROOT = r"D:\learning-neural-antiderivatives\models"
    MODE = "DoG-Blur"   # or "DoG-Noblur"
    model_dir = os.path.join(BASE_ROOT, MODE, "1d")
    eval_dir = f"evaluation_analytic_1d_{MODE.lower()}"
    os.makedirs(eval_dir, exist_ok=True)

    valid_funcs = {"ackley", "gm", "hr"}
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    print(f"Found {len(model_files)} total models in {model_dir}")

    N = 2048
    x_vals = torch.linspace(-1, 1, N).view(-1, 1).cuda()
    x_np = x_vals.cpu().numpy()

    total_mse = 0.0
    count = 0
    logs = []

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        print(f"\n== Evaluating: {model_file} ==")

        func_match = re.match(r"(ackley|gm|hr)", model_file.lower())
        if not func_match:
            print(f"Skipping {model_file} — not a valid analytic function.")
            continue
        func_name = func_match.group(1)

        order_matches = re.findall(r"order[_=]?([0-9]+)", model_file)
        order = int(order_matches[-1]) if order_matches else 0

        # Extract scale
        scale_match = re.search(r"scale[_=]?([0-9.]+)", model_file)
        if scale_match:
            scale_str = scale_match.group(1).rstrip("._")
            try:
                scale = float(scale_str)
            except ValueError:
                print(f"Invalid scale value '{scale_str}' in {model_file}, defaulting to 0.01")
                scale = 0.01
        else:
            scale = 0.01


        print(f"Function: {func_name}, Order: {order}, Scale: {scale}")

        try:
            model = load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        try:
            pred = chunked_derivative(model, x_vals, order).reshape(-1, 1)
            gt = get_ground_truth(func_name, x_np).reshape(-1, 1)
        except Exception as e:
            print(f"Error evaluating {model_file}: {e}")
            continue

        if order == 0:
            pred = pred * -scale
        elif order == 1:
            pred = pred * (scale ** 2)
        elif order == 2:
            pred = pred * -(scale ** 3)

        mse = np.mean((pred - gt) ** 2)
        total_mse += mse
        count += 1
        print(f"MSE = {mse:.8e}")

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(x_np, gt, label="Ground Truth", linewidth=1)
        plt.plot(x_np, pred, '--', label=f"Pred (Order {order}, scale={scale})", linewidth=1)
        plt.title(f"{func_name.upper()} | Order={order}, Scale={scale}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f"{func_name}_order{order}_scale{scale}.png"))
        plt.close()

        logs.append(f"{model_file}, func={func_name}, order={order}, scale={scale}, MSE={mse:.8e}\n")

    summary_path = os.path.join(eval_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(logs)

    avg_mse = total_mse / max(1, count)
    print(f"Done — {count} valid models evaluated.")
    print(f"Average MSE = {avg_mse:.8e}")
    print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    evaluate_analytic_models()
