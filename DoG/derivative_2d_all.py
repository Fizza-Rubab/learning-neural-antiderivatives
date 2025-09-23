import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from model import CoordinateNet_ordinary as CoordinateNet
from torch.func import vmap, jacfwd, jacrev
from ismael.images.image_io import send_to_tev
from ismael.images.image_io import load_image, save_image
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
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.stack([grid_y, grid_x], dim=-1)
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
    x = 1
    for i in range(0, coords.shape[0], chunk_size):

        if i % 150 == 0:
            print(x)
        chunk = coords[i:i + chunk_size]
        chunk.requires_grad_(True)
        out = nth_derivative(model, chunk, order)
        outputs.append(out.detach().cpu())
        x+=1
    return torch.cat(outputs, dim=0).numpy()


def evaluate_model_old(net_path, image_path, order):
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


    H, W, _ = gt.shape
    coords = build_2d_grid(H, W)

    pred = chunked_derivative(model, coords, order)
    pred = pred.reshape(H, W, 3)
    mse = np.mean((pred - gt) ** 2)

    return pred, gt, mse



def evaluate_model(net_path, image_path, order, lpips_fn, scale, blur= False):
    weights = torch.load(net_path)
    model = CoordinateNet(
        weights['output'], weights['activation'], weights['input'],
        weights['channels'], weights['layers'], weights['encodings'],
        weights['normalize_pe'], weights['pe'], norm_exp=0
    ).cuda()
    model.load_state_dict(weights['ckpt'])
    model.eval()

    if blur:
        gt = np.load(image_path, allow_pickle=True).item()['res']
    else:
        gt = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        gt = gt.astype(np.float32) / 255.0
        gt = pad_image(gt)

    gt = gt[::2, ::2]

    H, W, _ = gt.shape
    coords = build_2d_grid(H, W)

    pred = chunked_derivative(model, coords, order)
    pred = pred.reshape(H, W, 3)

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
    ckpt_root = "/HPS/antiderivative_project/work/Autoint/experiments/results_2d"
    eval_dir = "evaluation_2d"
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    mse_log_path = os.path.join(eval_dir, "mse_results.txt")
    with open(mse_log_path, 'w') as f:
        for img_file in os.listdir(image_dir):
            if not img_file.endswith(".png"):
                continue

            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(image_dir, img_file)

            for order in [1, 2]:
                ckpt_path = os.path.join(ckpt_root, f"Autoint_{base_name}_order={order}", "current.pth")
                if not os.path.exists(ckpt_path):
                    print(f"Missing: {ckpt_path}")
                    continue

                try:
                    pred, gt, mse = evaluate_model(ckpt_path, img_path, order)
                except Exception as e:
                    print(f"Error on {base_name} order {order}: {e}")
                    continue

                # Save plot
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                axes[0].imshow(pred)
                axes[0].set_title(f"Predicted (order={order})")
                axes[1].imshow(gt)
                axes[1].set_title("Ground Truth")
                for ax in axes:
                    ax.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{base_name}_order{order}.png"))
                plt.close()

                log = f"{base_name}, order={order}, MSE={mse:.6f}\n"
                print(log.strip())
                f.write(log)



def main3():

    # order 0 without blur
    #image_dir = "/HPS/n_ntumba/work/Fizza_project/data/images/"
    #ckpt_root = "/HPS/n_ntumba/work/Fizza_project/Experiment_2d_without_blur/1st_order/"
    # eval_dir = '/HPS/n_ntumba/work/Fizza_project/Experiment_2d_without_blur/visual'

    image_dir = "/HPS/n_ntumba/work/Fizza_project/data/image_mc_order=2/"
    ckpt_root = "/HPS/n_ntumba/work/Fizza_project/Experiment2d_with_blur/1st_order"
    eval_dir = '/HPS/n_ntumba/work/Fizza_project/Experiment2d_with_blur/visuals'

    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    save_exrs = eval_dir

    mse_log_path = os.path.join(eval_dir, "mse_results.txt")

    ckpt_dirs = os.listdir(ckpt_root)
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    blur = True



    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    count = 0
    for i in range(len(ckpt_dirs)):
        count += 1
        current_ckpt = ckpt_dirs[i]
        order = 1
        scale = 0.01
        base_name = current_ckpt.split("_")[2]


        if blur:
            img_path = os.path.join(image_dir, f'{base_name}_2d_order_2_gaussian_0.03_samples_100000.npy')
        else:
            img_path = os.path.join(image_dir, f'{base_name}.png')

        ckpt_path = os.path.join(ckpt_root, current_ckpt, f'current.pth')


        pred, gt, mse, psnr, ssim, lpips_val = evaluate_model(ckpt_path, img_path, order, lpips_fn, scale, blur)  # order-1



        # mse = ((pred - gt) ** 2).mean()
        total_mse += mse
        total_psnr += psnr
        total_ssim += ssim
        total_lpips += lpips_val

        print(f'base_name: {base_name}, order: {order}, mse: {mse:.6f}, psnr: {psnr:.6f}, ssim: {ssim:.6f}, lpips: {lpips_val:.6f}')



        save_image(pred, os.path.join(save_exrs, f'{base_name}.exr'))
        save_image(gt, os.path.join(save_exrs, f'{base_name}_ref.exr'))

        ##send_to_tev('prediction', pred)
        #send_to_tev('GT reference', gt)
        #exit()



        # Save plot
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(pred)
        axes[0].set_title(f"Predicted (order={order})")
        axes[1].imshow(gt)



        axes[1].set_title("Ground Truth")
        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_name}_order{order}_scale_{scale}.png"))
        plt.close()

        log = f"{base_name}, order={order}_scale_{scale}, MSE={mse:.6f}\n"
        # print(log.strip())
        #f.write(log)

    print(f"Total mse: {total_mse:.6f}")
    print(f"Total psnr: {total_psnr:.6f}")
    print(f"Total ssim: {total_ssim:.6f}")
    print(f"Total lpips: {total_lpips:.6f}")

    print()
    print(f'count : {count}')
    print(f'average mse: {total_mse/count:.6f}')
    print(f'average psnr: {total_psnr/count:.6f}')
    print(f'average ssim: {total_ssim/count:.6f}')
    print(f'average lpips: {total_lpips/count:.6f}')



def main2():

    image_dir = "/HPS/antiderivative_project/work/data/images"
    ckpt_root = "/HPS/antiderivative_project/work/Autoint/experiments/results_2d"
    eval_dir = '/HPS/n_ntumba/work/Fizza_project/Experiment_2d_without_blur/plots2/multiplication by the factor'

    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    save_exrs = eval_dir

    mse_log_path = os.path.join(eval_dir, "mse_results.txt")
    with open(mse_log_path, 'w') as f:

        # DoG_blur_0008_2d_order_2_gaussian_0.03_samples_100000_2d_scale_0.01_order_0
        # ckpt_paths = '/HPS/n_ntumba/work/Fizza_project/Minimal_kernel_experiment/'
        ckpt_paths = '/HPS/n_ntumba/work/Fizza_project/Experiment_2d_without_blur/1st_order'
        # ckpt_paths = '/HPS/n_ntumba/work/Fizza_project/test_debias/ckpt/'
        models = os.listdir(ckpt_paths)

        for i in range(len(models)):

            current_Model = models[i]

            scale = current_Model.split("_")[5]
            order = int(current_Model.split("_")[-1]) + 1

            print(order)
            # exit()
            # scale = current_Model.split("_")[12]
            # order = int(current_Model.split("_")[-1]) + 1

            base_name = 'temp2d'
            img_path = os.path.join(image_dir, '0008.png')

            ckpt_path = os.path.join(ckpt_paths, current_Model, f'current.pth')
            print(ckpt_path)


            pred, gt, mse = evaluate_model(ckpt_path, img_path, order)  # order-1
            send_to_tev(f'pred_temp2', pred * (float(scale) ** 2))
            exit()

            # if order-1 == 0:
            #     print(f'factor : {(float(scale) ** 2)}')
            #     pred = pred / (float(scale) ** 2)
            # elif order-1 == 1:
            #     pred = pred / (((float(scale) ** 2) ** 2))
            # elif order-1 == 2:
            #     pred = pred / (((float(scale) ** 2) ** 3))

            if order - 1 == 0:
                print(f'factor : {(float(scale) ** 2)}')
                pred = pred * (float(scale) ** 2)
            elif order - 1 == 1:
                pred = pred * (((float(scale) ** 2) ** 2))
            elif order - 1 == 2:
                pred = pred * (((float(scale) ** 2) ** 3))

            save_image(pred, os.path.join(save_exrs, f'oder_{order}_scale_{scale}_{i}.exr'))
            save_image(gt, os.path.join(save_exrs, f'Reference.exr'))


            # print(pred.shape)
            # print(gt.shape)
            # send_to_tev('prediction', pred)
            # send_to_tev('GT reference', gt)
            #continue

            #exit()


            # Save plot
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(pred)
            axes[0].set_title(f"Predicted (order={order})")
            axes[1].imshow(gt)



            axes[1].set_title("Ground Truth")
            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{base_name}_order{order}_scale_{scale}_{i}.png"))
            plt.close()

            log = f"{base_name}, order={order}_scale_{scale}, MSE={mse:.6f}\n"
            print(log.strip())
            f.write(log)



def isolate():

    order = 1
    scale = 0.01

    if order - 1 == 0:
        pred = (float(scale) ** 2)
    elif order - 1 == 1:
        pred = (((float(scale) ** 2) ** 2))
    elif order - 1 == 2:
        pred = (((float(scale) ** 2) ** 3))

    print(pred)


    order = 0
    sigma = 0.01
    if order == 0:
        kernel = sigma ** 2
    elif order == 1:
        kernel = ((sigma ** 2) ** 2)
    elif order == 2:
        kernel = ((sigma ** 2) ** 3)

    print(kernel)


if __name__ == "__main__":
    main3()
    # isolate()
