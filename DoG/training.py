import torch
import sys
from utilities import calculate_psnr
import time
# from ismael.images.image_io import send_to_tev
from utilities import compute_old_mixed_conv_gauss



def get_budget(dim, order):
    if dim == 1:
        if order == 0:
            return 2
        elif order == 1:
            return 3
        else:
            return 4

    elif dim == 2:
        if order == 0:
            return 4
        elif order == 1:
            return 9
        else:
            return 16

    elif dim == 3:
        if order == 0:
            return 8
        elif order == 1:
            return 27
        else:
            return 64

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
        interpolator_fn, sobol=None):
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

    budget = get_budget(args.in_channel, args.order)
    print()
    print(f'Running budget without BLur : {budget}')
    print(f'IN put dimension', args.in_channel)
    print(f'Order of Running', args.order)
    print()

    # Model related DoG Related informations
    # -----------------------------------------------------------------------------------------
    print(f'Using stratified sampling : {args.strata}')
    if args.in_channel == 1:

        # 1d case ----------
        sigma = args.kernel_scale
        order = args.order  # 1 or 2
        radius = sigma * 7
        antithetic = get_antithetic(order)
        budget = get_budget(args.in_channel, order) if args.budget == 0 else args.budget
        factor = 2 if antithetic else 1
        pdf = (2 * radius)
        samples2 = budget // factor if antithetic else budget

    elif args.in_channel == 2:
        # 2d case ---------
        sigma = args.kernel_scale
        order = args.order  # 1 or 2
        radius = sigma * 7
        antithetic = get_antithetic(order)
        budget = get_budget(args.in_channel, order)
        factor = 4 if antithetic else 1
        pdf = (2 * radius) ** 2
        samples2 = budget // factor if antithetic else budget

    # todo i am here doing this.
    elif args.in_channel == 3:
        # 3 case ----------
        sigma = args.kernel_scale
        order = args.order  # 1 or 2
        radius = sigma * 7
        antithetic = get_antithetic(order)
        budget = get_budget(args.in_channel, order)
        factor = 8 if antithetic else 1
        pdf = (2 * radius) ** 3
        samples2 = budget // factor if antithetic else budget

    print('\n-----------------------------------------------------------------------------------------')
    print(f'regime we are in')
    print(f'sigma', sigma)
    print(f'order', order)
    print(f'radius', radius)
    print(f'antithetic', antithetic)
    print(f'budget', budget)
    print(f'factor', factor)
    print(f'samples2', pdf)
    print(f'samples2', samples2)
    print('-----------------------------------------------------------------------------------------\n')

    # -----------------------------------------------------------------------------------------

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

        convolution_output = convolution_fn(args,
                                            model,
                                            radius,
                                            antithetic,
                                            input_tensor,
                                            factor,
                                            pdf,
                                            samples2,
                                            sigma,
                                            order, sobol)

        # print(f'in the training loop')
        # print(convolution_output.shape)
        # print(monte_carlo_rgb.shape)
        # exit()

        # convolution_output = convolution_fn(model,
        #                                     input_tensor,
        #                                     control_pts_coords,
        #                                     control_pts_vals,
        #                                     control_pts_nums,
        #                                     args)

        # print("convolution_output", convolution_output.shape)

        if args.debias == 1:
            convolution_output2 = convolution_fn(args,
                                                model,
                                                radius,
                                                antithetic,
                                                input_tensor,
                                                factor,
                                                pdf,
                                                samples2,
                                                sigma,
                                                order, sobol)

            loss1 = (convolution_output.float() - monte_carlo_rgb.float())
            loss2 = (convolution_output2.float() - monte_carlo_rgb.float()).detach()


            loss = (loss1 * loss2).mean()
        else:
            loss = loss_fn(convolution_output.float(), monte_carlo_rgb.float())

        # print(loss.min(), loss.max())

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
    print(f"Total training time: {(et - st):.6f}", flush=True)
    # ------------------------------------------------------------------------------------------------------------------


def get_antithetic(order):
    if order == 0:
        return True
    elif order == 1:
        return False
    elif order == 2:
        return True

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
