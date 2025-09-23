import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities import mesh_to_sdf_tensor, save_mesh
from model import CoordinateNet_ordinary as CoordinateNet
from torch.func import vmap, jacfwd, jacrev
import time
from ismael.images.image_io import send_to_tev


def pad_sdf(sdf_volume, pad_fraction=0.3, constant_value=1.0):
    d, h, w = sdf_volume.shape[:3]
    pd, ph, pw = int(d * pad_fraction), int(h * pad_fraction), int(w * pad_fraction)
    padding = ((pd, pd), (ph, ph), (pw, pw))
    if sdf_volume.ndim == 4:
        padding += ((0, 0),)
    return np.pad(sdf_volume, padding, mode='constant', constant_values=constant_value)


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





def evaluate_model_sdf(net_path, mesh_path, order, size=256, chunk_size=4096, blur=False, scale=0.1):
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights['pe'], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    if blur:
        gt = np.load(mesh_path, allow_pickle=True).item()['res'][..., 0]
        gt = gt[::4, ::4, ::4]
    else:
        gt = mesh_to_sdf_tensor(mesh_path, size).astype(np.float32)
        gt = pad_sdf(gt)
        gt = gt[::4, ::4, ::4]


    gt = torch.from_numpy(gt).cuda()
    D, H, W = gt.shape

    coords = build_3d_grid(D, H, W)
    pred = []
    x = 1
    for i in range(0, coords.shape[0], chunk_size):
        print(x)
        chunk = coords[i:i+chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order+1)
        pred.append(out.detach().cpu())
        x+=1
    pred = torch.cat(pred, dim=0).view(D, H, W)

    if order == 0:
        pred = pred / -scale ** 3
    elif order == 1:
        pred = pred / ((scale ** 3) ** 2)
    elif order == 2:
        pred = pred / -((scale ** 3) ** 3)

    mse = torch.mean((pred - gt.cpu()) ** 2).item()
    return pred.numpy(), gt.cpu().numpy(), mse


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


def main():

    log_lines = []
    # ckpt_folder = '/HPS/n_ntumba/work/Fizza_project/Experiment3d_without_blur/1st_order/'
    # eval_dir = "/HPS/n_ntumba/work/Fizza_project/Experiment3d_without_blur/visuals"
    # gt_folder = "/HPS/n_ntumba/work/Fizza_project/data/geometry"

    eval_dir = '/HPS/n_ntumba/work/Fizza_project/Experiment3d _with_blur/visual'
    ckpt_folder = "/HPS/n_ntumba/work/Fizza_project/Experiment3d _with_blur/1st_order"
    gt_folder = "/HPS/n_ntumba/work/Fizza_project/data/geometry_mc_order=2/"

    # pilot study
    eval_dir = '/HPS/n_ntumba/work/Fizza_project/pilot_study/3d_first_order/files2'
    ckpt_folder = "/HPS/n_ntumba/work/Fizza_project/pilot_study/3d_first_order/ckpt"
    gt_folder = "/HPS/n_ntumba/work/Fizza_project/data/geometry"

    blur = False

    plot_dir = os.path.join(eval_dir, "plots")
    mesh_out_dir = os.path.join(eval_dir, "meshes")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(mesh_out_dir, exist_ok=True)

    folders = os.listdir(ckpt_folder)
    #files_gt = os.listdir(mesh_path)
    #print(folders)
    #print(files_gt)


    order = 0
    scale = 0.2

    total_mse = 0
    count = 0
    for i in range(len(folders)):
        count += 1

        current_model = folders[i]
        scale = float(current_model.split('_')[-3])
        current_file = os.path.join(ckpt_folder, current_model)
        # base_name = f'{current_model.split("_")[2]}'
        base_name = 'ABC_00002074'

        ckpt_path = os.path.join(current_file, "current.pth")
        #scale = float(scale)
        mesh_path = glob.glob(f'{gt_folder}/*{base_name}*')[0]

        if not os.path.isfile(ckpt_path):
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue

        pred, gt, mse = evaluate_model_sdf(ckpt_path, mesh_path, order, chunk_size=4096, blur=blur, scale=scale)
        total_mse += mse

        print(f'base_name : {base_name}: scale {scale} current_mse', mse)

        # Save image slice
        slice_path = os.path.join(plot_dir, f"{base_name}_order_{order}_{scale}.png")
        plot_sdf_slice(pred, gt, z_idx=pred.shape[0] // 2, save_name=slice_path)

        # Save mesh
        os.makedirs(os.path.join(mesh_out_dir, f"{base_name}_order{order}_{scale}_{i}"), exist_ok=True)
        try:
            save_mesh(pred, os.path.join(mesh_out_dir, f"{base_name}_order{order}_{scale}_{i}"))
        except Exception as e:
            print("Mesh couldn't be saved:", e)

        # Prepare log entry
        ###line = f"{base_name}__{scale}, order={order}, MSE={mse:.6f}\n"
        #print(line.strip())
        #log_lines.append(line)
        #print(time.time() - st, "elapsed", flush=True)

        # Write all logs at the end
        #mse_log_path = os.path.join(eval_dir, "mse_results.txt")
        #with open(mse_log_path, 'a') as f:
         #   f.writelines(log_lines)

    print(f'total mse: {total_mse}')
    print(f'count: {count}')
    print(f'average psnr: {total_mse / count:.6f}')




def main2():
    mesh_dir = "/HPS/antiderivative_project/work/data/geometry"
    ckpt_root = "/HPS/antiderivative_project/work/Autoint/experiments/results_3d"
    eval_dir = "evaluation_3d"
    plot_dir = os.path.join(eval_dir, "plots")
    mesh_out_dir = os.path.join(eval_dir, "meshes")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(mesh_out_dir, exist_ok=True)

    log_lines = []

    for mesh_file in os.listdir(mesh_dir):
        if not mesh_file.endswith(".ply"):
            continue

        base_name = os.path.splitext(mesh_file)[0]
        mesh_path = os.path.join(mesh_dir, mesh_file)

        for order in [2]:
            print(f"File: {base_name}, Order: {order}", flush=True)
            st = time.time()
            ckpt_path = os.path.join(ckpt_root, f"Autoint_{base_name}_order={order}", "current.pth")
            if not os.path.isfile(ckpt_path):
                print(f"Skipping missing checkpoint: {ckpt_path}")
                continue

            try:
                pred, gt, mse = evaluate_model_sdf(ckpt_path, mesh_path, order, chunk_size=4096)
            except Exception as e:
                print(f"Error on {base_name} order {order}: {e}")
                continue

            # Save image slice
            slice_path = os.path.join(plot_dir, f"{base_name}_order{order}.png")
            plot_sdf_slice(pred, gt, z_idx=pred.shape[0] // 2, save_name=slice_path)

            # Save mesh
            os.makedirs(os.path.join(mesh_out_dir, f"{base_name}_order{order}"), exist_ok=True)
            try:
                save_mesh(pred, os.path.join(mesh_out_dir, f"{base_name}_order{order}"))
            except Exception as e:
                print("Mesh couldn't be saved:", e)

            # Prepare log entry
            line = f"{base_name}, order={order}, MSE={mse:.6f}\n"
            print(line.strip())
            log_lines.append(line)
            print(time.time() - st, "elapsed", flush=True)

    # Write all logs at the end
    mse_log_path = os.path.join(eval_dir, "mse_results.txt")
    with open(mse_log_path, 'a') as f:
        f.writelines(log_lines)




def temp():

    path = '/HPS/n_ntumba/work/Fizza_project/data/geomc_metry_mc_order=2/'
    meshes = glob.glob(os.path.join(path, "*.npy"))

    for i in range(len(meshes)):
        data = np.load(meshes[i], allow_pickle=True).item()['res'][..., 0]
        print(f'data shape: {data.shape}')
        slice = data[data.shape[0] // 2]
        send_to_tev(f'slice_{i}', slice)

    # save_mesh(data, save)


if __name__ == "__main__":

    # temp()
    main()
