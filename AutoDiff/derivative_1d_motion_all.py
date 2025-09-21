import os
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.func import vmap, jacfwd, jacrev

from model import CoordinateNet_ordinary as CoordinateNet


def pad_signal_1d(signal, pad_fraction=0.3):
    pad = int(signal.shape[0] * pad_fraction)
    return np.pad(signal, ((pad, pad), (0, 0)), mode='reflect')


def normalize_array(x, out_min=0, out_max=1):
    in_min, in_max = np.min(x), np.max(x)
    return (out_max - out_min) / (in_max - in_min) * (x - in_min) + out_min


def read_pose(file_path, normalize=True, num_frames=5000):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("Skeletool")]

    pose_list = [np.array(list(map(float, line.split()[1:]))).reshape(-1, 3) for line in lines]
    pose_array = np.stack(pose_list)

    if num_frames is not None:
        total = pose_array.shape[0]
        if num_frames < total:
            start = (total - num_frames) // 2
            pose_array = pose_array[start:start + num_frames]
        elif num_frames > total:
            raise ValueError(f"Only {total} frames available.")

    if normalize:
        pose_array = normalize_array(pose_array)
    return pose_array


def nth_derivative(model, x, order):
    if order == 1:
        return vmap(jacfwd(model))(x)
    elif order == 2:
        return vmap(jacrev(jacfwd(model)))(x)
    elif order == 3:
        return vmap(jacfwd(jacrev(jacfwd(model))))(x)
    else:
        raise ValueError("Only orders 1–3 supported.")


def chunked_derivative(model, coords, order, chunk_size=2048):
    outputs = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def evaluate_model(net_path, pose_path, order):
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights["pe"], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    gt = read_pose(pose_path).reshape(-1, 69)
    N = gt.shape[0]
    x = torch.linspace(-1, 1, N).view(-1, 1).cuda()

    pred = chunked_derivative(model, x, order).reshape(-1, 69)
    mse = np.mean((pred - gt) ** 2)
    return x.detach().cpu().numpy(), pred, gt, mse


def main():
    pose_dir = "/HPS/antiderivative_project/work/data/poses"
    ckpt_root = "/HPS/antiderivative_project/work/Autoint/experiments/results_1d"
    eval_dir = "evaluation_1d"
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    orders = [1, 2, 3]
    all_logs = []

    for order in orders:
        print(f"\n== Evaluating Order {order} ==")
        for fname in os.listdir(pose_dir):
            if not fname.endswith(".txt"):
                continue

            base_name = os.path.splitext(fname)[0]
            pose_path = os.path.join(pose_dir, fname)
            ckpt_path = os.path.join(ckpt_root, f"Autoint_{base_name}_order={order}", "current.pth")

            if not os.path.exists(ckpt_path):
                print(f"Skipping missing checkpoint: {ckpt_path}")
                continue

            try:
                x_vals, pred, gt, mse = evaluate_model(ckpt_path, pose_path, order)
            except Exception as e:
                print(f"Error evaluating {fname} order {order}: {e}")
                continue

            print(f"{base_name}, order={order}, MSE={mse:.6f}")
            all_logs.append(f"{base_name}, order={order}, MSE={mse:.6f}\n")

            joint_idx = 5  
            plt.figure(figsize=(10, 4))
            plt.plot(x_vals, gt[:, joint_idx], label="GT", linewidth=1)
            plt.plot(x_vals, pred[:, joint_idx], '--', label="Pred", linewidth=1)
            plt.xlabel("Time")
            plt.ylabel("Pose Signal")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f"{base_name}_order{order}.png")
            plt.savefig(plot_path)
            plt.close()


            npy_dir = os.path.join(eval_dir, "npys")
            os.makedirs(npy_dir, exist_ok=True)	
            npy_file = os.path.join(npy_dir, f"{base_name}_order{order}.npy")
            np.save(npy_file, {'x_vals': x_vals, 'pred': pred, 'gt': gt})
            # break

    # Write log file once
    os.makedirs(eval_dir, exist_ok=True)
    log_path = os.path.join(eval_dir, "mse_results.txt")
    with open(log_path, 'w') as f:
        f.writelines(all_logs)


if __name__ == "__main__":
    main()
