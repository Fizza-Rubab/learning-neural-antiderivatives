import torch
import numpy as np

def map_range(values, old_range, new_range):
    NewRange = (new_range[0] - new_range[1])
    OldRange = (old_range[0] - old_range[1])
    new_values = (((values - old_range[0]) * NewRange) / OldRange) + new_range[0]
    return new_values

def do_1d_conv(model,
               x_samples,
               ctrl_pts_coords,
               ctrl_vals,
               num_ctrl_pts,
               args):
    output_dims = 1
    sample_xs = x_samples
    duplicated_xs = torch.repeat_interleave(sample_xs, num_ctrl_pts, dim=1)[..., None]
    coordinates_reshaped = ctrl_pts_coords[None, :, None]
    convolution_coordinates = duplicated_xs + coordinates_reshaped
    B = sample_xs.shape[0]
    convolution_coordinates = convolution_coordinates.view(-1, 1)
    integral_values = model(convolution_coordinates)
    integral_values = integral_values.view(B, num_ctrl_pts, output_dims)
    diracs = ctrl_vals[None, :, None]
    diracs = torch.repeat_interleave(diracs, output_dims, dim=-1)
    convolved_results = (integral_values * diracs).sum(1)
    return convolved_results


def do_1d_motion_conv(model,
                      x_samples,
                      ctrl_pts_coords,
                      ctrl_vals,
                      num_ctrl_pts,
                      args,
                      order=0,
                      scale=1):
    output_dims = 69
    sample_xs = x_samples
    duplicated_xs = torch.repeat_interleave(sample_xs, num_ctrl_pts, dim=1)[..., None]
    coordinates_reshaped = ctrl_pts_coords[None, :, None]
    convolution_coordinates = duplicated_xs + coordinates_reshaped
    B = sample_xs.shape[0]
    convolution_coordinates = convolution_coordinates.view(-1, 1)
    integral_values = model(convolution_coordinates)

    integral_values = integral_values.view(B, num_ctrl_pts, output_dims)


    if order == 0:
        integral_values = integral_values * -scale
    elif order == 1:
        integral_values = integral_values * scale ** 2
    elif order == 2:
        integral_values = integral_values * -scale ** 3


    diracs = ctrl_vals[None, :, None]
    diracs = torch.repeat_interleave(diracs, output_dims, dim=-1)
    convolved_results = (integral_values * diracs).sum(1)
    return convolved_results


def do_2d_conv(model,
               xy_samples,
               ctrl_pts_coords,
               ctrl_vals,
               num_ctrl_pts,
               args,
               order=0,
               scale=1):
    output_dims = 3
    samples = xy_samples
    sample_xs = samples[:, :1]
    sample_ys = samples[:, 1:]

    coordinates, values = ctrl_pts_coords, ctrl_vals
    num_ctrl_pts = num_ctrl_pts

    duplicated_xs = torch.repeat_interleave(sample_xs, num_ctrl_pts, dim=1)[..., None]
    duplicated_ys = torch.repeat_interleave(sample_ys, num_ctrl_pts, dim=1)[..., None]

    duplicated_grid = torch.cat([duplicated_xs, duplicated_ys], -1)

    coordinates_reshaped = coordinates[None, ...]
    convolution_coordinates = duplicated_grid + coordinates_reshaped

    # ------------------------------------------------------------------------------------------------------------------

    # sampling the mlp
    integral_values = model(convolution_coordinates)

    if order == 0:
        integral_values = integral_values * (float(scale) ** 2)


    diracs = values[None, :, None]
    diracs = torch.repeat_interleave(diracs, output_dims, dim=-1)

    # ------------------------------------------------------------------------------------------------------------------

    convolved_results = (integral_values * diracs).sum(1)
    return convolved_results


def do_3d_conv(model,
               x_grid,
               ctrl_pts_coords,
               ctrl_vals,
               num_ctrl_pts,
               args, order=0,
                      scale=1):
    samples = x_grid
    sample_xs = samples

    coordinates, values = ctrl_pts_coords, ctrl_vals
    num_ctrl_pts = num_ctrl_pts

    duplicated_xs = torch.repeat_interleave(sample_xs[:, None, :], num_ctrl_pts, dim=1)
    coordinates_reshaped = coordinates[None]
    convolution_coordinates = duplicated_xs + coordinates_reshaped

    # ------------------------------------------------------------------------------------------------------------------

    # sampling the mlp
    integral_values = model(convolution_coordinates)

    if order == 0:
        integral_values = integral_values / -scale ** 3


    diracs = values[None, :, None]

    # ------------------------------------------------------------------------------------------------------------------

    convolved_results = (integral_values * diracs).sum(1)
    return convolved_results


def do_video_conv(model,
                  sample_nums_torch,
                  kernel_control_points,
                  kernel_values,
                  n_control_points,
                  args,order=0,
                      scale=1):

    coordinates, values = kernel_control_points, kernel_values  # kernel_object.get_control_points()
    num_ctrl_pts = n_control_points

    samples = sample_nums_torch
    sample_xs = samples[:, :1]
    sample_ys = samples[:, 1:2]
    sample_ts = samples[:, 2:]

    coordinates_reshaped = coordinates[None, :, None]
    duplicated_xs = torch.repeat_interleave(sample_xs, num_ctrl_pts, dim=1)[..., None]
    duplicated_ys = torch.repeat_interleave(sample_ys, num_ctrl_pts, dim=1)[..., None]
    duplicated_ts = torch.repeat_interleave(sample_ts, num_ctrl_pts, dim=1)[..., None]
    duplicated_ts = duplicated_ts + coordinates_reshaped

    duplicated_grid = torch.cat([duplicated_xs, duplicated_ys, duplicated_ts], -1)
    convolution_coordinates_torch = duplicated_grid

    sampled_video = model(convolution_coordinates_torch.view(-1, 3))
    sampled_video = sampled_video.view(-1, num_ctrl_pts, 3)

    diracs = values[None, :, None]

    convolved_results = sampled_video * diracs
    convolved_results = convolved_results.sum(1)

    return convolved_results


# todo add the dog function here for 1d 2d and 3d


def gaussian_function(x, mu=0.0, sigma=1.0):
    return 1.0 / (sigma * (2.0 * np.pi) ** 0.5) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def dv1_gauss(x, sigma):
    grad_of_gauss = -(x / sigma ** 2) * gaussian_function(x, mu=0.0, sigma=sigma)
    return grad_of_gauss


def dv2_gauss(x, sigma):
    term1 = x ** 2 / sigma ** 4
    term2 = 1 / (sigma ** 2)
    grad_of_gauss = (term1 - term2) * gaussian_function(x, mu=0.0, sigma=sigma)
    return grad_of_gauss


def dv3_gauss(x, sigma):
    term1 = -(x ** 3 / sigma ** 6)
    term2 = (3 * x) / (sigma ** 4)
    grad_of_gauss = (term1 + term2) * gaussian_function(x, mu=0.0, sigma=sigma)
    return grad_of_gauss


def sample_gaussian_derivative_2d(x, sigma, order, antithetic):

    if antithetic:
        _x = x[..., 0:1]
        _y = x[..., 1:]
        x = torch.cat([c(_x, _y), c(-_x, _y), c(_x, -_y), -c(_x, _y)], 1)

    if order == 0:
        vals = dv1_gauss(x, sigma)
        vals = vals[..., 0:1] * vals[..., 1:]

    elif order == 1:
        vals = dv2_gauss(x, sigma)
        vals = vals[..., 0:1] * vals[..., 1:]

    elif order == 2:
        vals = dv3_gauss(x, sigma)
        vals = vals[..., 0:1] * vals[..., 1:]

    return vals, x


def sample_gaussian_derivative_3d(x, sigma, order, antithetic):
    if antithetic:
        _x = x[..., 0:1]
        _y = x[..., 1:2]
        _z = x[..., 2:]

        x = torch.cat([c2(_x, _y, _z), c2(-_x, _y, _z),
                              c2(_x, -_y, _z), c2(_x, _y, -_z), c2(-_x, -_y, _z),
                              c2(-_x, _y, -_z), c2(_x, -_y, -_z), -c2(_x, _y, _z)
                       ], dim=1)

    if order == 0:
        vals = dv1_gauss(x, sigma)
        vals = vals[..., 0:1] * vals[..., 1:2] * vals[..., 2:]

    elif order == 1:
        vals = dv2_gauss(x, sigma)
        vals = vals[..., 0:1] * vals[..., 1:2] * vals[..., 2:]

    elif order == 2:
        vals = dv3_gauss(x, sigma)
        vals = vals[..., 0:1] * vals[..., 1:2] * vals[..., 2:]

    return vals, x


def sample_gaussian_derivative_1d(x, sigma, order, antithetic):

    if antithetic:
        x = torch.cat([x, -x], 1)

    if order == 0:
        vals = dv1_gauss(x, sigma)
    elif order == 1:
        vals = dv2_gauss(x, sigma)
    elif order == 2:
        vals = dv3_gauss(x, sigma)

    return vals, x


def c(x, y):
    return torch.concatenate([x, y], -1)


def c2(x, y, z):
    return torch.concatenate([x, y, z], -1)


def do_1d_gaussian_dv_conv(args, model, radius, antithetic, coords, factor, pdf, samples2, sigma, order, sobol=None):
    # args, model, RAD2D, RES, antithetic, coords, factor, pdf, samples2, sigma, order

    if args.strata == 1:
        if order == 1:
            # _2d_coords = torch.linspace(-radius, radius, samples2).float().cuda()
            # cell_size = (_2d_coords[1] - _2d_coords[0]).item()
            # _2d_coords = _2d_coords.view(-1, 1)[None]
            # _2d_coords = torch.repeat_interleave(_2d_coords, coords.shape[0], 0)
            # jitters = torch_uniform_sample(-cell_size/2, cell_size/2, _2d_coords.shape).float().cuda()
            # _2d_coords = _2d_coords + jitters

            _2d_coords = map_range(sobol.draw(samples2 * coords.shape[0]).view(coords.shape[0], samples2, 1), (0, 1), (-radius, radius)).float().cuda()

    else:
        _2d_coords = torch.from_numpy(np.random.uniform(-radius, radius, (coords.shape[0], samples2, 1))).float().cuda()

    kernel, _2d_coords = sample_gaussian_derivative_1d(_2d_coords, sigma, order, antithetic)


    if args.order == 0:
        kernel = kernel * sigma
    elif args.order == 1:
        kernel = kernel * sigma ** 2
    elif args.order == 2:
        kernel = kernel * sigma ** 3



    # print(_2d_coords.shape, _2d_coords.min(), _2d_coords.max())
    # print(kernel.shape)
    # print(_2d_coords.shape)
    # print(coords.shape)
    # print()


    input_coordinates = torch.repeat_interleave(coords[..., 0:1][:, None], samples2 * factor, 1) + _2d_coords
    #print(input_coordinates.shape)
    output = model(input_coordinates.float())
    #print(f'net output : {output.shape}')



    # print(input_coordinates.shape)
    # print(output.shape)
    # print(kernel.shape)
    # print()

    #print(output.shape)
    #print(kernel.shape)
    #print(pdf)

    output_ = (output * kernel) * pdf
    output = output_.mean(1)

    #print(f'mean output : {output.shape}')


    return output

def torch_uniform_sample(low, high, shape):
    return torch.rand(*shape) * (high - low) + low



def do_2d_gaussian_dv_conv(args, model, radius, antithetic, coords, factor, pdf, samples2, sigma, order, sobol=None):
    # print(f'inside derivative of gaussian')
    # print(radius)
    # print(antithetic)
    # print(coords.shape)
    # print(factor)
    # print(pdf)
    # print(samples2)
    # print(sigma)
    # print(order)
    # print()



    if args.strata == 1:
        if order == 1:
            xs = torch.linspace(-radius, radius, int(np.sqrt(samples2))).float().cuda()
            cell_size = (xs[1] - xs[0]).item()
            _2d_coords = torch.stack(torch.meshgrid(xs, xs), -1).view(-1, 2)[None]
            _2d_coords = torch.repeat_interleave(_2d_coords, coords.shape[0], 0)

            jitters = torch_uniform_sample(0, cell_size, _2d_coords.shape).float().cuda()
            _2d_coords = _2d_coords + jitters

    else:
        _2d_coords = torch.from_numpy(np.random.uniform(-radius, radius, (coords.shape[0], samples2, 2))).cuda()

    kernel, _2d_coords = sample_gaussian_derivative_2d(_2d_coords, sigma, order, antithetic)
    kernel = kernel.cuda()


    # print(kernel.min())
    # print(kernel.max())
    # print()

    if args.order == 0:
         kernel = kernel * sigma ** 2
    elif args.order == 1:
         kernel = kernel * ((sigma ** 2) ** 2)
    elif args.order == 2:
         kernel = kernel * ((sigma ** 2) ** 3)


    # print(kernel.min())
    # print(kernel.max())
    # exit()


    _2d_coords = _2d_coords.cuda()

    # print(kernel.min())
    # print(kernel.max())
    # exit()




    # print(_2d_coords.shape)
    # print(kernel.shape)
    # print()

    repeated_x = torch.repeat_interleave(coords[..., 0:1][:, None], samples2 * factor, 1) + _2d_coords[..., 0:1]
    repeated_y = torch.repeat_interleave(coords[..., 1:][:, None], samples2 * factor, 1) + _2d_coords[..., 1:]
    input_coordinates = torch.cat([repeated_x, repeated_y], -1).float()


    # print(repeated_x.shape)
    # print(repeated_y.shape)
    # print(input_coordinates.shape)
    # print()

    output = model(input_coordinates)
    output_ = (output * kernel) * pdf
    output = output_.mean(1)
    return output


def do_3d_gaussian_dv_conv(args, model, radius, antithetic, coords, factor, pdf, samples2, sigma, order, sobol=None):


    _2d_coords = torch.from_numpy(np.random.uniform(-radius, radius, (coords.shape[0], samples2, 3))).cuda()
    kernel, _2d_coords = sample_gaussian_derivative_3d(_2d_coords, sigma, order, antithetic)
    kernel = kernel.cuda()
    _2d_coords = _2d_coords.cuda()

    if args.order == 0:
         kernel = kernel * sigma ** 3
    elif args.order == 1:
         kernel = kernel * ((sigma ** 3) ** 2)
    elif args.order == 2:
         kernel = kernel * ((sigma ** 3) ** 3)


    # print(_2d_coords.shape)
    # print(kernel.shape)
    # print()

    repeated_x = torch.repeat_interleave(coords[..., 0:1][:, None], samples2 * factor, 1) + _2d_coords[..., 0:1]
    repeated_y = torch.repeat_interleave(coords[..., 1:2][:, None], samples2 * factor, 1) + _2d_coords[..., 1:2]
    repeated_z = torch.repeat_interleave(coords[..., 2:][:, None], samples2 * factor, 1) + _2d_coords[..., 2:]

    # print(repeated_x.shape)
    # print(repeated_y.shape)
    # print(repeated_z.shape)
    # print()

    input_coordinates = torch.cat([repeated_x, repeated_y, repeated_z], -1).float()


    output = model(input_coordinates)
    output_ = (output * kernel) * pdf
    output = output_.mean(1)

    return output


def calc_gauss(x, mu=0.0, sigma=1.0):
    return 1.0 / (sigma * (2.0 * np.pi) ** 0.5) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def grad_of_gaussiankernel(x, sigma):
    grad_of_gauss = -(x / sigma ** 2) * calc_gauss(x, mu=0.0, sigma=sigma)
    return grad_of_gauss


def compute_old_mixed_conv_gauss(args, direction, radius, input_tensor, model, mc_samples,
                                 antithetic=False,
                                 precision=32):
    def c(x, y):
        return torch.concatenate([x, y], -1)

    radius_loc = radius
    sigma_loc = radius_loc / 3
    original_mc_samples = mc_samples
    mc_samples = mc_samples  # * 1 if antithetic else mc_samples

    # print(radius_loc)
    # print(sigma_loc)
    # print(mc_samples)
    # exit()

    _2d_coords = (-radius_loc - radius_loc) * torch.rand(input_tensor.shape[0], mc_samples, 2).cuda() + radius_loc
    _x = _2d_coords[..., 0:1]
    _y = _2d_coords[..., 1:]
    _2d_coords = torch.cat([c(_x, _y), c(-_x, _y), c(_x, -_y), -c(_x, _y)], 1)

    duplicated_spatial = torch.repeat_interleave(input_tensor[:, None], mc_samples * 4, dim=1)
    shifted_spatial = duplicated_spatial + _2d_coords

    pdf = ((2 * radius_loc) ** 2)

    kernel_vals = grad_of_gaussiankernel(_2d_coords, sigma_loc)
    full_kernel = kernel_vals[..., 0:1] * kernel_vals[..., 1:]

    input_coords = shifted_spatial.double() if precision == 64 else shifted_spatial.float()
    output = model(input_coords)
    result = (output * full_kernel) * pdf
    result = result.mean(1)

    return result


