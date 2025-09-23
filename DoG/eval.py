import glob
import os.path
import time
import imageio
import matplotlib.pyplot as plt
import torch
import numpy as np
from utilities import minimal_kernel_diracs, TempKernel2d, TempKernel3d, TempKernel1d
from model import CoordinateNet_ordinary as CoordinateNet
from utilities import do_2d_conv, do_3d_conv, do_video_conv, do_1d_conv, do_1d_motion_conv
import sys
from functools import reduce
import click
from utilities import save_mesh, send_to_tev
from utilities import create_or_recreate_folders
import time
import soundfile as sf
import re
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
import lpips
import torch.nn.functional as F


def save_frames(frames, path):
    for i in range(frames.shape[2]):

        if i % 10 == 0:
            print(f'saved : {i}')

        if i < 10:
            filename = f'000{i}.png'
        elif 10 <= i < 100:
            filename = f'00{i}.png'
        elif 100 <= i < 1000:
            filename = f'0{i}.png'
        else:
            filename = i

        to_save = (np.clip(frames[..., i, :], 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(path, filename), to_save)


def load_diracs(path, scale, modality=1, order=1):
    diracs = torch.load(path)
    vals = diracs['ckpt']
    coords = diracs['ctrl_pts']

    if modality == 0:
        values = vals
        kernel_object = TempKernel1d()
    elif modality == 1:
        values = np.outer(vals, vals)
        coords = np.stack(np.meshgrid(coords, coords), -1)
        kernel_object = TempKernel2d()
    elif modality == 2 or modality == 3:
        values = np.outer(vals, vals)
        values = np.outer(values.ravel(), vals).reshape(len(vals), len(vals), len(vals))
        coords = np.stack(np.meshgrid(coords, coords, coords, indexing='ij'), -1)
        kernel_object = TempKernel3d()

    kernel_object.initialize_control_points(coords, values, order)
    kernel_object.shrink_kernel(scale)

    return kernel_object


def create_minimal_kernel_1d(order, half_size=1.0):
    diracs_x, diracs_y = minimal_kernel_diracs(order, half_size)
    kernel = TempKernel1d()
    kernel.initialize_control_points(diracs_x, diracs_y, order)
    return kernel


def create_minimal_kernel_3d(order, size):
    kernel_xs, kernel_ys = minimal_kernel_diracs(order, size)
    values = reduce(np.multiply.outer, (kernel_ys, kernel_ys, kernel_ys))
    coord_vals = np.stack(np.meshgrid(kernel_xs, kernel_xs, kernel_xs), -1)
    kernel = TempKernel3d()
    kernel.initialize_control_points(coord_vals, values, order)
    return kernel


def create_minimal_filter_2d(order, half_size=1.0):
    diracs_x, diracs_y = minimal_kernel_diracs(order, half_size)
    grid = np.stack(np.meshgrid(diracs_x, diracs_x), -1)
    values = np.outer(diracs_y, diracs_y)
    kernel = TempKernel2d()  # Kernel2d()
    kernel.initialize_control_points(grid, values, order)
    return kernel


def load_network(net_path,
                 shape,
                 precision=32,
                 modality=1):
    weights = torch.load(net_path)
    model = CoordinateNet(weights['output'],
                          weights['activation'],
                          weights['input'],
                          weights['channels'],
                          weights['layers'],
                          weights['encodings'],
                          weights['normalize_pe'],
                          10,
                          norm_exp=2).cuda()

    # load the weights into the network
    model.load_state_dict(weights['ckpt'])
    model = model.eval()
    model = model.double() if precision == 64 else model.float()
    # ------------------------------------------------------------------------------------------------------------------

    # generate coordinates to sample
    if modality == 0:
        coords_x = np.linspace(-1, 1, shape[0], endpoint=True)
        xy_grid = coords_x[..., None]  # shape (N, 1)
        convolution_tensor = np.zeros((xy_grid.shape[0], 69))  # Mono audio
    elif modality == 1:
        coords_x = np.linspace(-1, 1, shape[0], endpoint=True)
        coords_y = np.linspace(-1, 1, shape[1], endpoint=True)
        xy_grid = np.stack(np.meshgrid(coords_x, coords_y, indexing='ij'), -1)
        convolution_tensor = np.zeros((xy_grid.shape[0], xy_grid.shape[1], 3))
    else:
        coords_x = np.linspace(-1, 1, shape[0], endpoint=True)
        coords_y = np.linspace(-1, 1, shape[1], endpoint=True)
        coords_z = np.linspace(-1, 1, shape[2], endpoint=True)
        xy_grid = np.stack(np.meshgrid(coords_x, coords_y, coords_z, indexing='ij'), -1)

        if modality == 2:
            convolution_tensor = np.zeros((xy_grid.shape[0], xy_grid.shape[1], xy_grid.shape[2], 1))
        else:
            convolution_tensor = np.zeros((xy_grid.shape[0], xy_grid.shape[1], xy_grid.shape[2], 3))

    xy_grid = torch.from_numpy(xy_grid).float().contiguous().cuda()
    xy_grid = xy_grid.double() if precision == 64 else xy_grid.float()

    return model, xy_grid.float(), np.float32(convolution_tensor)


def evaluate(kern_path,
             net_path,
             shape,
             kernel_scale,
             conv_fn,
             precision=32,
             block_size=32,
             modality=1,
             order=1,
             scale=1):

    if precision == 32:
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)

    # ------------------------------------------------------------------------------------------------------------------
    if modality == 0:
        kernel_object = create_minimal_kernel_1d(order, 1 / kernel_scale)
        # kernel_object = load_diracs(kern_path, kernel_scale, modality, order)

    elif modality == 1:
        kernel_object = create_minimal_filter_2d(order, 1 / kernel_scale)
        # kernel_object = load_diracs(kern_path, kernel_scale, modality, order)
    elif modality == 2:
        kernel_object = create_minimal_kernel_3d(order, 1 / kernel_scale)
        # kernel_object = load_diracs(kern_path, kernel_scale, modality, order)
    else:
        kernel_object = create_minimal_kernel_1d(order, 1 / kernel_scale)

    model, xy_grid_torch, convolution_tensor = load_network(net_path, shape, 32, modality)
    model = model.eval()

    # ------------------------------------------------------------------------------------------------------------------

    control_pts_coords, control_pts_vals = kernel_object.get_control_points()
    control_pts_coords = torch.from_numpy(control_pts_coords).cuda().float()
    control_pts_vals = torch.from_numpy(control_pts_vals).cuda().float()
    control_pts_nums = kernel_object.get_n_control_points()

    # ------------------------------------------------------------------------------------------------------------------
    sub_block = block_size

    with torch.no_grad():
        if modality == 0:
            for i in range(0, xy_grid_torch.shape[0], sub_block):
                current_grid = xy_grid_torch[i:i + sub_block]  # shape (B, 1)
                cL, _ = current_grid.shape  # Length and 1 (since it's 1D)

                current_output = conv_fn(model,
                                         current_grid.contiguous().view(-1, 1),
                                         control_pts_coords,
                                         control_pts_vals,
                                         control_pts_nums,
                                         None,
                                         order, scale)

                current_output = current_output.view(cL, convolution_tensor.shape[-1])
                convolution_tensor[i:i + sub_block] = current_output.cpu().numpy()




        elif modality == 1:
            for i in range(0, xy_grid_torch.shape[0], sub_block):
                for j in range(0, xy_grid_torch.shape[1], sub_block):
                    current_grid = xy_grid_torch[i:i + sub_block, j:j + sub_block]
                    cH, cW, _ = current_grid.shape
                    current_output = conv_fn(model,
                                             current_grid.contiguous().view(-1, 2),
                                             control_pts_coords,
                                             control_pts_vals,
                                             control_pts_nums,
                                             None, order, scale)

                    current_output = current_output.view(cH, cW, 3)
                    convolution_tensor[i:i + sub_block, j:j + sub_block] = current_output.cpu().numpy()





        else:
            # Existing 3D and video branch (unchanged)
            for i in range(0, xy_grid_torch.shape[0], sub_block):
                for j in range(0, xy_grid_torch.shape[1], sub_block):
                    for k in range(0, xy_grid_torch.shape[2], sub_block):
                        current_grid = xy_grid_torch[i:i + sub_block, j:j + sub_block, k:k + sub_block]
                        cH, cW, cD, _ = current_grid.shape
                        current_output = conv_fn(model,
                                                 current_grid.contiguous().view(-1, 3),
                                                 control_pts_coords,
                                                 control_pts_vals,
                                                 control_pts_nums,
                                                 None,
                                         order, scale)

                        current_output = current_output.view(cH, cW, cD, convolution_tensor.shape[-1])
                        convolution_tensor[i:i + sub_block, j:j + sub_block,
                        k:k + sub_block] = current_output.cpu().numpy()



    # ------------------------------------------------------------------------------------------------------------------
    return convolution_tensor


@click.command()
@click.option("--model_path", default='', help="path to data")
@click.option("--kernel_path", default='', help="path to kernel")
@click.option("--save_path", default='', help="path to save output")
@click.option("--modality", default=1, help="modality flag")
@click.option("--width", default=128, help="signal width")
@click.option("--height", default=128, help="signal height")
@click.option("--depth", default=100, help="signal depth (3d and video)")
@click.option("--block_size", default=32, help="sub block size")
@click.option("--kernel_scale", default=20.0, help="sub block size")
@click.option("--order", default=20.0, help="order")
def run_evaluation(model_path,
                   kernel_path,
                   save_path,
                   modality,
                   width,
                   height,
                   depth,
                   block_size,
                   kernel_scale,
                   order):
    save_dir = save_path

    path1 = '/HPS/n_ntumba/work/Fizza_project/Experiment1d_without_blur/1st'
    path2 = '/HPS/n_ntumba/work/Fizza_project/Experiment1d_with_blur/1st'
    path3 = '/HPS/n_ntumba/work/Fizza_project/Experiment1d_without_blur/2nd'
    path4 = '/HPS/n_ntumba/work/Fizza_project/Experiment1d_with_blur/2nd/'
    path5 = '/HPS/n_ntumba/work/Fizza_project/Experiment_2d_without_blur/1st_order'
    path6 = '/HPS/n_ntumba/work/Fizza_project/Experiment2d_with_blur/1st_order'
    path7 = '/HPS/n_ntumba/work/Fizza_project/Experiment3d_without_blur/1st_order'
    path8 = '/HPS/n_ntumba/work/Fizza_project/Experiment3d _with_blur/1st_order'

    tag = 2
    # all paths contain multiple models.
    model_paths = [path1, path2, path3, path4, path5, path6, path7, path8][-tag:]
    print(model_paths)
    #exit()

    block_size = 32
    modality_array = [0, 0, 0, 0, 1, 1, 2, 2][-tag:]

    type = ['1d_motion_without_blur_order0',
            '1d_motion_with_blur_order0',
            '1d_motion_without_blur_order1',
            '1d_motion_with_blur_order1',
            '2d_without_blur_order0',
            '2d_with_blur_order0',
            '3d_without_blur_order0',
            '3d_with_blur_order0', ][-tag:]

    orders = [0,
             0,
             1,
             1,
             0,
             0,
             0,
             0][-tag:]

    _3d = 10
    scales = [0.01, 0.01, 0.03, 0.03, 0.01, 0.01, 0.2, 0.2][-tag:]
    wi = [5000, 5000, 5000, 5000, 1638, 1638, _3d, _3d][-tag:]
    he = [0, 0, 0, 0, 1638, 1638, _3d, _3d][-tag:]
    de = [0, 0, 0, 0, 0, 0, _3d, _3d][-tag:]


    for i in range(len(model_paths)):
        current_model = model_paths[i]
        save_path = os.path.join(save_dir, type[i])
        create_or_recreate_folders(save_path)

        modality = modality_array[i]
        width = wi[i]
        height = he[i]
        depth = de[i]
        block_size = block_size
        active_scale = scales[i]
        order = orders[i]

        model_lists = os.listdir(current_model)

        for j in range(len(model_lists)):

            current_model_path = os.path.join(current_model, model_lists[j], f'current.pth')
            model_name = os.path.basename(os.path.dirname(current_model_path))

            kernel_scales = [1 / 0.1, 1 / 0.2, 1 / 0.3]
            for kernel_scale in kernel_scales:
                path = current_model_path
                kern_path = kernel_path

                conv_fn = do_2d_conv

                if modality == 0:
                    conv_fn = do_1d_motion_conv
                elif modality == 1:
                    conv_fn = do_2d_conv
                elif modality == 2:
                    conv_fn = do_3d_conv
                elif modality == 3:
                    conv_fn = do_video_conv

                st = time.time()

                output_tensor = evaluate(kern_path,
                                         path,
                                         (width, height, depth),
                                         kernel_scale,
                                         conv_fn,
                                         modality=modality,
                                         block_size=block_size,
                                         order=order,
                                         scale=active_scale)
                et = time.time()
                print("elapsed time", et - st, "s")

                if modality == 0:
                    mse_log = defaultdict(list)
                    padding_fraction = 0.3
                    start = int(padding_fraction * 5000)

                    clipped_tensor = output_tensor[start:-start]
                    clipped_tensor = output_tensor[start:-start]
                    subject = int(re.findall(r'\d+', model_name)[0])
                    print("model_name", model_name)
                    gt_path = fr"../convolution_mc/motion/subject_{subject}_motion1d_order_{int(order)}_minimal_{np.round(1 / kernel_scale, 1)}_samples_100000.npy"
                    print("gt_path", gt_path)
                    print("order", order)
                    gt_np = np.load(gt_path, allow_pickle=True).item()['res']
                    mse = ((gt_np[start:-start, :] - output_tensor[start:-start, :]) ** 2).mean()
                    print("MSE", mse)
                    mse_log[(order, round(1 / kernel_scale, 1))].append(mse)
                    plt.figure()
                    plt.plot(gt_np[start:-start, 11], label="gt")
                    plt.plot(output_tensor[start:-start, 11], label="pred")
                    plt.legend()
                    print(f'other')
                    print(os.path.join(save_path, f"{model_name}_{np.round(1 / kernel_scale, 1)}.png"))
                    #print(save_path)
                    #print(f'here')

                    plt.savefig(os.path.join(save_path, f"{model_name}_{np.round(1 / kernel_scale, 1)}.png"))

                    summary_path = os.path.join(save_path, "results_summary.txt")

                    with open(summary_path, "a") as f:
                        f.write(f"{model_name}, kernel_scale={1 / kernel_scale:.2f}, order={order}, MSE={mse:.8f}\n")

                    print(f"Appended results to {summary_path}")

                # images
                if modality == 1:
                    padding_fraction = 0.3
                    start = int(padding_fraction * 1024)
                    clipped_pred = output_tensor[start:-start, start:-start, :]
                    name = (re.findall(r'\d+', model_name)[0])
                    gt_path = fr"../convolution_mc/image_mc_order={int(order)}/{name}_2d_order_{int(order)}_minimal_{np.round(1 / kernel_scale, 1)}_samples_200000.npy"
                    gt_np = np.load(gt_path, allow_pickle=True).item()['res']
                    gt_crop = gt_np[start:-start, start:-start, :]

                    raw_mse = ((output_tensor[start:-start, start:-start, :] - gt_crop) ** 2).mean()

                    pred_clip = np.clip(clipped_pred, 0, 1)
                    gt_clip = np.clip(gt_crop, 0, 1)
                    ssim_val = ssim(pred_clip, gt_clip, channel_axis=-1, data_range=1.0)
                    lpips_model = lpips.LPIPS(net='alex').cuda()
                    pred_torch = torch.from_numpy(pred_clip.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
                    gt_torch = torch.from_numpy(gt_clip.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
                    lpips_val = lpips_model(pred_torch.cuda(), gt_torch.cuda()).item()

                    combined = np.concatenate([gt_clip, pred_clip], axis=1)
                    to_save = (combined * 255).astype(np.uint8)
                    img_path = os.path.join(save_path, f"{name}_order={order}_{np.round(1 / kernel_scale, 1)}.png")
                    imageio.imwrite(img_path, to_save)
                    print(f"MSE={raw_mse:.8f}, SSIM={ssim_val:.8f}, LPIPS={lpips_val:.8f}\n")
                    summary_path = os.path.join(save_path, "results_summary.txt")
                    with open(summary_path, "a") as f:
                        f.write(
                            f"{model_name}, kernel_scale={1 / kernel_scale:.2f}, order={order}, "
                            f"MSE={raw_mse:.8f}, SSIM={ssim_val:.8f}, LPIPS={lpips_val:.8f}\n"
                        )

                    print(f"Saved comparison and metrics to {summary_path}")

                if modality == 2:
                    padding_fraction = 0.3
                    start = int(padding_fraction * 256)
                    clipped_pred = output_tensor[start:-start, start:-start, start:-start]

                    name = model_name.split("_3d")[0][9:]
                    print(name)
                    gt_path = glob.glob(os.path.join(f'../convolution_mc/geometry_mc_order={int(order)}', '*_'))
                    print(gt_path)
                    print(f'here')
                    gt_path = f"../convolution_mc/geometry_mc_order={int(order)}/{name}_3d_order_{int(order)}_{np.round(1 / kernel_scale, 1)}_samples_20000.npy"

                    print(os.listdir(f'../convolution_mc/geometry_mc_order={int(order)}'))
                    print(gt_path)
                    exit()
                    gt_np = np.load(gt_path, allow_pickle=True).item()['res']
                    gt_crop = gt_np[start:-start, start:-start, start:-start]

                    mse = ((gt_crop - clipped_pred) ** 2).mean()
                    print(f"MSE (SDF): {mse:.8f}")
                    mesh_name = f"{model_name}_order{order}_scale{np.round(1 / kernel_scale, 1)}.ply"
                    try:
                        save_mesh(clipped_pred, save_path, mesh_name)
                        # save_mesh(clipped_pred, save_path, 'gt_' + mesh_name)
                        print(f"Saved mesh: {mesh_name}")
                    except:
                        print("No surface found")
                    summary_path = os.path.join(save_path, "results_summary.txt")
                    with open(summary_path, "a") as f:
                        f.write(f"{model_name}, kernel_scale={1 / kernel_scale:.2f}, order={order}, MSE={mse:.8f}\n")
                    print(f"Appended results to {summary_path}")


                # Videos
                elif modality == 3:
                    save_frames(output_tensor, save_path)


if __name__ == '__main__':
    run_evaluation()
