import torch
import sys
from utilities import calculate_psnr
import time

def train(
        SAVE_PATH,
        args,
        model,
        optim,
        scheduler,
        writer,
        net_dictionary,
        kernel_object,
        monte_carlo_np,
        convolution_fn,
        sampling_fn,
        loss_fn,
        interpolator_fn):
    # ------------------------------------------------------------------------------------------------------------------
    global_iteration = 0
    if args.init_ckpt is not None:
        checkp = torch.load(args.init_ckpt)
        global_iteration = checkp['epoch']

    # ------------------------------------------------------------------------------------------------------------------
    model = model.train()

    # ------------------------------------------------------------------------------------------------------------------
    control_pts_coords, control_pts_vals = kernel_object.get_control_points()
    control_pts_coords = torch.from_numpy(control_pts_coords).cuda().float()
    control_pts_vals = torch.from_numpy(control_pts_vals).cuda().float()
    control_pts_nums = kernel_object.get_n_control_points()
    st = time.time()
    for step in range(args.num_steps + 1):

        global_iteration += 1
        batch_size = args.batch
        optim.zero_grad()

        if global_iteration % 10000 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = global_iteration
            net_dictionary['optim'] = optim.state_dict()
            torch.save(net_dictionary, SAVE_PATH + f'/checkpoint_{global_iteration}.pth')

        if global_iteration % 1000 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = global_iteration
            net_dictionary['optim'] = optim.state_dict()
            torch.save(net_dictionary, SAVE_PATH + f'/current.pth')

        # data sampling
        # ----------------------------------------------------------------------------------------------------------
        input_tensor, monte_carlo_rgb = sampling_fn(args, interpolator_fn, monte_carlo_np)

        # convolution
        # ----------------------------------------------------------------------------------------------------------
        convolution_output = convolution_fn(model,
                                            input_tensor,
                                            control_pts_coords,
                                            control_pts_vals,
                                            control_pts_nums,
                                            args)
        loss = loss_fn(convolution_output.float(), monte_carlo_rgb.float())
        # ----------------------------------------------------------------------------------------------------------

        loss.backward()
        optim.step()

        # ----------------------------------------------------------------------------------------------------------
        if global_iteration % 200 == 0:
            print(f'Iteration : {global_iteration},'
                  f' train loss:, {loss.item()},'
                  f' Batch Size:, {batch_size},'
                  f'kernel : {1 / args.kernel_scale}', flush=True)

        # ----------------------------------------------------------------------------------------------------------

        writer.add_scalar('Integral Loss', loss.item(), global_iteration)
        scheduler.step()
        sys.stdout.flush()

    net_dictionary['ckpt'] = model.state_dict()
    net_dictionary['epoch'] = global_iteration
    net_dictionary['optim'] = optim.state_dict()
    torch.save(net_dictionary, SAVE_PATH + f'/model_final.pth')

    
    et = time.time()
    print(f"Total training time: {(et -st):.6f}", flush=True)

    # ------------------------------------------------------------------------------------------------------------------


# def train(
#         SAVE_PATH,
#         args,
#         model,
#         optim,
#         scheduler,
#         writer,
#         net_dictionary,
#         kernel_object,
#         monte_carlo_np,
#         convolution_fn,
#         sampling_fn,
#         loss_fn,
#         interpolator_fn):
    
#     global_iteration = 0
#     if args.init_ckpt is not None:
#         checkp = torch.load(args.init_ckpt)
#         global_iteration = checkp['epoch']

#     model = model.train()

#     control_pts_coords, control_pts_vals = kernel_object.get_control_points()
#     control_pts_coords = torch.from_numpy(control_pts_coords).cuda().float()
#     control_pts_vals = torch.from_numpy(control_pts_vals).cuda().float()
#     control_pts_nums = kernel_object.get_n_control_points()

#     training_time_total = 0.0
#     final_psnr = 0.0

#     PSNR_EVAL_INTERVAL = 1000  # set to 1000 steps

#     for step in range(args.num_steps + 1):
#         global_iteration += 1
#         batch_size = args.batch

#         # --- overhead timing for logging/checkpoints ---
#         overhead_start = time.time()

#         if global_iteration % 10000 == 0:
#             net_dictionary['ckpt'] = model.state_dict()
#             net_dictionary['epoch'] = global_iteration
#             net_dictionary['optim'] = optim.state_dict()
#             torch.save(net_dictionary, SAVE_PATH + f'/checkpoint_{global_iteration}.pth')

#         if global_iteration % 1000 == 0:
#             net_dictionary['ckpt'] = model.state_dict()
#             net_dictionary['epoch'] = global_iteration
#             net_dictionary['optim'] = optim.state_dict()
#             torch.save(net_dictionary, SAVE_PATH + f'/current.pth')

#         overhead_end = time.time()

#         # --- training time (measured) ---
#         train_start = time.time()

#         optim.zero_grad()
#         input_tensor, monte_carlo_rgb = sampling_fn(args, interpolator_fn, monte_carlo_np)
#         convolution_output = convolution_fn(model,
#                                             input_tensor,
#                                             control_pts_coords,
#                                             control_pts_vals,
#                                             control_pts_nums,
#                                             args)

#         loss = loss_fn(convolution_output.float(), monte_carlo_rgb.float())
#         loss.backward()
#         optim.step()

#         train_end = time.time()
#         training_time_total += train_end - train_start

#         # --- Logging ---
#         if global_iteration % 200 == 0:
#             print(f'Iteration: {global_iteration}, Loss: {loss.item():.6f}, PSNR: {final_psnr:.2f}, '
#                   f'Training Time: {training_time_total:.2f}s')

#         writer.add_scalar('Integral Loss', loss.item(), global_iteration)
#         scheduler.step()
#         sys.stdout.flush()

#         # --- PSNR check ---
#         if global_iteration % PSNR_EVAL_INTERVAL == 0:
#             with torch.no_grad():
#                 model.eval()
#                 full_input, full_target = sampling_fn(args, interpolator_fn, monte_carlo_np, full=True)
#                 full_prediction = convolution_fn(model,
#                                                  full_input,
#                                                  control_pts_coords,
#                                                  control_pts_vals,
#                                                  control_pts_nums,
#                                                  args)

#                 final_psnr = calculate_psnr(full_prediction.float(), full_target.float())
#                 print(f"[PSNR Eval] Step {global_iteration}, PSNR = {final_psnr:.2f} dB")
#                 model.train()

#                 if final_psnr >= 30.0:
#                     print(f"✅ Early stopping: PSNR ≥ 30 at iteration {global_iteration}")
#                     break

#     # --- Save final model ---
#     net_dictionary['ckpt'] = model.state_dict()
#     net_dictionary['epoch'] = global_iteration
#     net_dictionary['optim'] = optim.state_dict()
#     net_dictionary['training_time_total'] = training_time_total
#     torch.save(net_dictionary, SAVE_PATH + f'/model_final.pth')

#     print(f"🔚 Final Iteration: {global_iteration}")
#     print(f"⏱️ Pure Training Time: {training_time_total:.2f} seconds")
#     print(f"📈 Final PSNR: {final_psnr:.2f} dB")