import torch


def do_1d_conv(model,
               x_samples,
               ctrl_pts_coords,
               ctrl_vals,
               num_ctrl_pts,
               order):
    output_dims = 1
    n = (order + 1) ** output_dims
    sample_xs = x_samples
    B = sample_xs.shape[0]
    duplicated_xs = torch.repeat_interleave(sample_xs, num_ctrl_pts, dim=1)[..., None]
    coordinates_reshaped = ctrl_pts_coords[None, :, None]
    convolution_coordinates = duplicated_xs + coordinates_reshaped  
    convolution_coordinates_flat = convolution_coordinates.view(-1, 1)
    integral_values = model(convolution_coordinates_flat)
    integral_values = integral_values.view(B, num_ctrl_pts, n)
    x = convolution_coordinates.view(B, num_ctrl_pts, 1)

    print("reshaped x:", x.shape)
    print("reshaped integral_values:", integral_values.shape)

    if order == 0:
        result = integral_values[:, :, 0:1] 
    elif order == 1:
        result = x * integral_values[:, :, 0:1] - integral_values[:, :, 1:2]
    elif order == 2:
        result = (
            0.5 * x ** 2 * integral_values[:, :, 0:1]
            - x * integral_values[:, :, 1:2]
            + 0.5 * integral_values[:, :, 2:3]
        )
    diracs = ctrl_vals[None, :, None]
    diracs = torch.repeat_interleave(diracs, output_dims, dim=-1)  
    convolved_results = (result * diracs).sum(1) 
    return convolved_results


def do_1d_motion_conv(model,
               x_samples,
               ctrl_pts_coords,
               ctrl_vals,
               num_ctrl_pts,
               order):
    output_dims = 1
    n = 69 * (order + 1) ** output_dims
    sample_xs = x_samples
    B = sample_xs.shape[0]
    duplicated_xs = torch.repeat_interleave(sample_xs, num_ctrl_pts, dim=1)[..., None]
    coordinates_reshaped = ctrl_pts_coords[None, :, None] 
    convolution_coordinates = duplicated_xs + coordinates_reshaped  
    B = sample_xs.shape[0]
    convolution_coordinates = convolution_coordinates.view(-1, 1)
    integral_values = model(convolution_coordinates)  
    integral_values = integral_values.view(B, num_ctrl_pts, n)
    x = convolution_coordinates.view(B, num_ctrl_pts, 1)
    if order == 0:
        result = integral_values[:, :, 0:69] 
    elif order == 1:
        result = x * integral_values[:, :, 0:69] - integral_values[:, :, 69:]
    elif order == 2:
        result = (
            0.5 * x ** 2 * integral_values[:, :, 0:69]
            - x * integral_values[:, :, 69:2*69]
            + 0.5 * integral_values[:, :, 2*69:]
        )
    diracs = ctrl_vals[None, :, None]
    diracs = torch.repeat_interleave(diracs, output_dims, dim=-1)  
    convolved_results = (result * diracs).sum(1) 
    return convolved_results





def do_2d_conv(model,
               xy_samples,
               ctrl_pts_coords,
               ctrl_vals,
               num_ctrl_pts,
               order):
    dims = 2
    output_dims = 3 
    n = output_dims * (order + 1) ** dims
    sample_xs = xy_samples[:, :1]
    sample_ys = xy_samples[:, 1:]
    B = xy_samples.shape[0]
    duplicated_xs = torch.repeat_interleave(sample_xs, num_ctrl_pts, dim=1)[..., None]
    duplicated_ys = torch.repeat_interleave(sample_ys, num_ctrl_pts, dim=1)[..., None]
    duplicated_grid = torch.cat([duplicated_xs, duplicated_ys], dim=-1)  
    coordinates_reshaped = ctrl_pts_coords[None, ...]
    convolution_coordinates = duplicated_grid + coordinates_reshaped  
    integral_values = model(convolution_coordinates.view(-1, 2)) 
    integral_values = integral_values.view(B, num_ctrl_pts, n)
    x = convolution_coordinates[:, :, :1]
    y = convolution_coordinates[:, :, 1:]
    if order == 0:
        basis = integral_values[:, :, 0:3]
    elif order == 1:
        basis = (
            x * y * integral_values[:, :, 0:3] -
            y * integral_values[:, :, 3:6] -
            x * integral_values[:, :, 6:9] + 
            integral_values[:, :, 9:12]
        )
    diracs = ctrl_vals[None, :, None] 
    diracs = torch.repeat_interleave(diracs, output_dims, dim=-1)
    convolved_results = (basis * diracs).sum(1)
    return convolved_results

def do_3d_conv(model,
               x_grid,
               ctrl_pts_coords,
               ctrl_vals,
               num_ctrl_pts,
               order):

    sample_xs = x_grid
    B = sample_xs.shape[0]

    duplicated_xs = torch.repeat_interleave(sample_xs[:, None, :], num_ctrl_pts, dim=1)  # (B, num_ctrl_pts, 3)
    coordinates_reshaped = ctrl_pts_coords[None]  # (1, num_ctrl_pts, 3)
    convolution_coordinates = duplicated_xs + coordinates_reshaped  # (B, num_ctrl_pts, 3)

    integral_values = model(convolution_coordinates)  # (B, num_ctrl_pts, n)

    if order == 1:
        x, y, z = convolution_coordinates[:, :, 0:1], convolution_coordinates[:, :, 1:2], convolution_coordinates[:, :, 2:3]
        integral_values = (
            x*y*z*integral_values[:, :, 0:1] - x*y*integral_values[:, :, 3:4]
            - x*z*integral_values[:, :, 2:3] + x*integral_values[:, :, 6:7]
            - y*z*integral_values[:, :, 1:2] + y*integral_values[:, :, 5:6]
            + z*integral_values[:, :, 4:5] - integral_values[:, :, 7:8]
        )
    elif order == 2:
        x, y, z = convolution_coordinates[:, :, 0:1], convolution_coordinates[:, :, 1:2], convolution_coordinates[:, :, 2:3]
        integral_values = (
            x**2*y**2*z**2/8 * integral_values[:, :, 0:1]
            - x**2*y**2*z/4 * integral_values[:, :, 5:6]
            + x**2*y**2/8 * integral_values[:, :, 6:7]
            - x**2*y*z**2/4 * integral_values[:, :, 3:4]
            + x**2*y*z/2 * integral_values[:, :, 15:16]
            - x**2*y/4 * integral_values[:, :, 17:18]
            + x**2*z**2/8 * integral_values[:, :, 4:5]
            - x**2*z/4 * integral_values[:, :, 16:17]
            + x**2/8 * integral_values[:, :, 18:19]
            - x*y**2*z**2/4 * integral_values[:, :, 1:2]
            + x*y**2*z/2 * integral_values[:, :, 11:12]
            - x*y**2/4 * integral_values[:, :, 13:14]
            + x*y*z**2/2 * integral_values[:, :, 7:8]
            - x*y*z/1 * integral_values[:, :, 19:20]
            + x*y/2 * integral_values[:, :, 22:23]
            - x*z**2/4 * integral_values[:, :, 9:10]
            + x*z/2 * integral_values[:, :, 21:22]
            - x/4 * integral_values[:, :, 25:26]
            + y**2*z**2/8 * integral_values[:, :, 2:3]
            - y**2*z/4 * integral_values[:, :, 12:13]
            + y**2/8 * integral_values[:, :, 14:15]
            - y*z**2/4 * integral_values[:, :, 8:9]
            + y*z/2 * integral_values[:, :, 20:21]
            - y/4 * integral_values[:, :, 24:25]
            + z**2/8 * integral_values[:, :, 10:11]
            - z/4 * integral_values[:, :, 23:24]
            + 1/8 * integral_values[:, :, 26:27]
        )
    diracs = ctrl_vals[None, :, None] 
    convolved_results = (integral_values * diracs).sum(1).sum(-1, keepdim=True) 
    return convolved_results


def do_video_conv(model,
                  sample_nums_torch,
                  kernel_control_points,
                  kernel_values,
                  n_control_points,
                  order):

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

