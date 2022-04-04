import numpy as np
import torch

from math import pi, cos, sin


def transform_points(points, transform, translate=True):
    # Append ones or zeros to get homogenous coordinates
    if translate:
        constant_term = np.ones_like(points[..., :1])
    else:
        constant_term = np.zeros_like(points[..., :1])
    points = np.concatenate((points, constant_term), axis=-1)

    points = np.einsum('nm,...m->...n', transform, points)
    return points[..., :3]


def frustum_cull(points, camera_pos, rays, near_plane=None, far_plane=None):
    corners = [rays[0, 0], rays[0, -1], rays[-1, -1], rays[-1, 0]]

    rel_points = points - np.expand_dims(camera_pos, 0)

    included = np.ones(points.shape[0])

    for i in range(4):
        c1 = corners[i]
        c2 = corners[(i+1) % 4]

        normal = np.cross(c1, c2)
        normal /= np.linalg.norm(normal)

        d = (rel_points * np.expand_dims(normal, 0)).sum(-1)

        included = np.logical_and(included, d >= 0)

    return included


def get_world_to_camera_matrix(camera_pos, vertical=None):
    # We assume that the camera is pointed at the origin
    camera_z = -camera_pos / torch.norm(camera_pos, dim=-1, keepdim=True)
    if vertical is None:
        vertical = torch.tensor((0., 0., 1.))
    else:
        vertical = torch.tensor(vertical)
    vertical = vertical.to(camera_pos)
    camera_x = torch.cross(camera_z, vertical.expand_as(camera_z), dim=-1)
    camera_x = camera_x / torch.norm(camera_x, dim=-1, keepdim=True)
    camera_y = torch.cross(camera_z, camera_x, dim=-1)

    camera_matrix = torch.stack((camera_x, camera_y, camera_z), -2)
    translation = -torch.einsum('...ij,...j->...i', camera_matrix, camera_pos)
    camera_matrix = torch.cat((camera_matrix, translation.unsqueeze(-1)), -1)
    return camera_matrix


def get_camera_to_world_matrix(camera_pos, vertical=None):
    world2cam = get_world_to_camera_matrix(camera_pos, vertical)
    return torch.inverse(world2cam)


def project_to_image_plane(camera_pos, x, focal_length=0.035, sensor_height=0.024, sensor_width=0.032):
    """
    Projects 3D world coordinates to 2D camera coordinates. Assumes
    that the camera is horizontal and pointed at the origin.
    Args:
        camera_pos: [N, 3] Camera positions.
        x: [N, P, 3] 3D points.
    Returns:
        [N, P, 2] 2D camera coordinates in the range [-0.5, 0.5].
    """
    camera_matrix = get_world_to_camera_matrix(camera_pos)
    camera_matrix = camera_matrix.unsqueeze(1).expand(list(x.shape) + [4])

    # Append ones to get homogeneous coordinates
    x = torch.cat((x, torch.ones_like(x[..., :1])), -1)

    cam_coords = torch.einsum('...ji,...i->...j', camera_matrix, x)

    img_x = (cam_coords[..., 0] / cam_coords[..., 2]) * focal_length
    img_y = (cam_coords[..., 1] / cam_coords[..., 2]) * focal_length

    img_x = img_x / sensor_width
    img_y = img_y / sensor_height

    img_coords = torch.stack((img_x, img_y), -1)
    return img_coords


def rotate_around_z_axis_np(points, theta):
    # Rotate point around the z axis
    results = np.zeros_like(points)
    results[..., 2] = points[..., 2]
    results[..., 0] = cos(theta) * points[..., 0] - sin(theta) * points[..., 1]
    results[..., 1] = sin(theta) * points[..., 0] + cos(theta) * points[..., 1]

    return results


def rotate_around_z_axis_torch(points, theta):
    # Rotate point around the z axis
    results = torch.zeros_like(points)
    results[..., 2] = points[..., 2]
    results[..., 0] = cos(theta) * points[..., 0] - sin(theta) * points[..., 1]
    results[..., 1] = sin(theta) * points[..., 0] + cos(theta) * points[..., 1]
    return results


def get_camera_rays(c_pos, width=320, height=240, focal_length=0.035, sensor_width=0.032, noisy=False,
                    vertical=None, c_track_point=None):
    #c_pos = np.array((0., 0., 0.))
    # The camera is pointed at the origin
    if c_track_point is None:
        c_track_point = np.array((0., 0., 0.))

    if vertical is None:
        vertical = np.array((0., 0., 1.))

    c_dir = (c_track_point - c_pos)
    c_dir = c_dir / np.linalg.norm(c_dir)

    img_plane_center = c_pos + c_dir * focal_length

    # The horizontal axis of the camera sensor is horizontal (z=0) and orthogonal to the view axis
    img_plane_horizontal = np.cross(c_dir, vertical)
    #img_plane_horizontal = np.array((-c_dir[1]/c_dir[0], 1., 0.))
    img_plane_horizontal = img_plane_horizontal / np.linalg.norm(img_plane_horizontal)

    # The vertical axis is orthogonal to both the view axis and the horizontal axis
    img_plane_vertical = np.cross(c_dir, img_plane_horizontal)
    img_plane_vertical = img_plane_vertical / np.linalg.norm(img_plane_vertical)

    # Double check that everything is orthogonal
    def is_small(x, atol=1e-7):
        return abs(x) < atol

    assert(is_small(np.dot(img_plane_vertical, img_plane_horizontal)))
    assert(is_small(np.dot(img_plane_vertical, c_dir)))
    assert(is_small(np.dot(c_dir, img_plane_horizontal)))

    # Sensor height is implied by sensor width and aspect ratio
    sensor_height = (sensor_width / width) * height

    # Compute pixel boundaries
    horizontal_offsets = np.linspace(-1, 1, width+1) * sensor_width / 2
    vertical_offsets = np.linspace(-1, 1, height+1) * sensor_height / 2

    # Compute pixel centers
    horizontal_offsets = (horizontal_offsets[:-1] + horizontal_offsets[1:]) / 2
    vertical_offsets = (vertical_offsets[:-1] + vertical_offsets[1:]) / 2

    horizontal_offsets = np.repeat(np.reshape(horizontal_offsets, (1, width)), height, 0)
    vertical_offsets = np.repeat(np.reshape(vertical_offsets, (height, 1)), width, 1)

    if noisy:
        pixel_width = sensor_width / width
        pixel_height = sensor_height / height
        horizontal_offsets += (np.random.random((height, width)) - 0.5) * pixel_width
        vertical_offsets += (np.random.random((height, width)) - 0.5) * pixel_height

    horizontal_offsets = (np.reshape(horizontal_offsets, (height, width, 1)) *
                          np.reshape(img_plane_horizontal, (1, 1, 3)))
    vertical_offsets = (np.reshape(vertical_offsets, (height, width, 1)) *
                        np.reshape(img_plane_vertical, (1, 1, 3)))

    image_plane = horizontal_offsets + vertical_offsets

    image_plane = image_plane + np.reshape(img_plane_center, (1, 1, 3))
    c_pos_exp = np.reshape(c_pos, (1, 1, 3))
    rays = image_plane - c_pos_exp
    ray_norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / ray_norms
    return rays.astype(np.float32)


def get_rays_from_intrinsics(intrinsics, height=720, width=1280, downsample=0):
    xmap, ymap = np.arange(width), np.arange(height)
    xmap, ymap = np.meshgrid(xmap, ymap)

    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    s = 1000.0
    points_x = (xmap - cx) / fx
    points_y = (ymap - cy) / fy
    points_z = np.ones_like(points_x)

    points = np.stack([points_x, points_y, points_z], axis=-1)
    for i in range(downsample):
        points = (points[0::2, 0::2] + points[1::2, 0::2] + points[0::2, 1::2] + points[1::2, 1::2]) / 4.
    return points.astype(np.float32)



def get_nerf_sample_points(camera_pos, rays, min_dist=0.035, max_dist=30, num_samples=256,
                           min_z=-0.1, deterministic=False):
    """
    Get uniform points for coarse NeRF sampling:

        Args:
            camera_pos: [..., 3] tensor indicating camera position
            rays: [..., 3] tensor indicating unit length directions of the pixel rays
            min_dist: focal length of the camera
            max_dist: maximum render distance
            num_samples: number of samples to generate
        Return:
            sample_depths: Depths of the sampled points, tensor of shape [..., num_samples]
            sample_points: 3d coordiantes of the sampled points, tensor of shape [..., num_samples, 3]
    """
    max_dist = torch.zeros_like(rays[..., 0]) + max_dist

    if min_z is not None:
        delta_z = min_z - camera_pos[..., 2]
        t_int = delta_z / rays[..., 2]
        t_int_clip = torch.logical_and(t_int >= 0., t_int <= max_dist)

        max_dist[t_int_clip] = t_int[t_int_clip]

    sample_segment_borders = torch.linspace(0., 1., num_samples+1).to(rays)
    while len(sample_segment_borders.shape) <= len(max_dist.shape):
        sample_segment_borders = sample_segment_borders.unsqueeze(0)
    sample_segment_borders = sample_segment_borders * (max_dist - min_dist).unsqueeze(-1) + min_dist

    if deterministic:
        sample_depths = (sample_segment_borders[..., 1:] + sample_segment_borders[..., :-1]) / 2
    else:
        sample_depth_dist = torch.distributions.Uniform(sample_segment_borders[..., :-1],
                                                        sample_segment_borders[..., 1:])
        sample_depths = sample_depth_dist.rsample()

    scaled_rays = rays.unsqueeze(-2) * sample_depths.unsqueeze(-1)
    sample_points = scaled_rays + camera_pos.unsqueeze(-2)
    return sample_depths, sample_points


def get_fine_nerf_sample_points(camera_pos, rays, depth_dist, depths,
                                min_dist=0.035, max_dist=30., num_samples=256,
                                deterministic=False):
    """
    Get points for fine NeRF sampling:

        Args:
            camera_pos: [..., 3] tensor indicating camera position
            rays: [..., 3] tensor indicating unit length directions of the pixel rays
            depth_dist: [..., s] tensor indicating the depth distribution obtained so far.
                Must sum to one along the s axis.
            depths: [..., s] tensor indicating the depths to which the depth distribution is referring.
            min_dist: focal length of the camera
            max_dist: maximum render distance
            num_samples: number of samples to generate
        Return:
            sample_depths: Depths of the sampled points, tensor of shape [..., s]
            sample_points: 3d coordiantes of the sampled points, tensor of shape [..., s, 3]
    """

    segment_borders = torch.cat((torch.zeros_like(depths[..., :1]) + min_dist,
                                 depths,
                                 1.5 * depths[..., -1:] - 0.5 * depths[..., -2:-1]), -1)
    histogram_weights = torch.zeros_like(segment_borders[..., 1:])

    # Allocate 75% of samples to previous segment, 0.25 to the following one.
    histogram_weights[..., :-1] = depth_dist * 0.75
    histogram_weights[..., 1:] += depth_dist * 0.25

    sample_depths = sample_pdf(segment_borders, histogram_weights, num_samples, deterministic=deterministic)
    scaled_rays = rays.unsqueeze(-2) * sample_depths.unsqueeze(-1)

    sample_points = scaled_rays + camera_pos.unsqueeze(-2)
    return sample_depths, sample_points


def sample_pdf(bins, depth_dist, num_samples, deterministic=False):
    """ Sample from histogram. Adapted from github.com/bmild/nerf

        bins: Boundaries of the histogram bins, shape [..., s+1]
        depth_dist: Probability of each bin, shape [..., s]. Must sum to one along the s axis.
        num_samples: Number of samples to collect from each histogram.
        deterministic: Whether to collect linearly spaced samples instead of sampling.
    """

    # Get pdf
    depth_dist += 1e-5  # prevent nans
    cdf = torch.cumsum(depth_dist, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1).contiguous()

    sample_shape = list(bins.shape[:-1]) + [num_samples]

    # Take uniform samples
    if deterministic:
        u = torch.linspace(0., 1., num_samples).to(bins)
        u = u.expand(sample_shape).contiguous()
    else:
        u = torch.rand(sample_shape).to(bins)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)

    inds_below = torch.maximum(torch.zeros_like(inds), inds-1)
    inds_above = torch.minimum(torch.zeros_like(inds) + (cdf.shape[-1]-1), inds)

    cdf_below = torch.gather(cdf, -1, inds_below)
    cdf_above = torch.gather(cdf, -1, inds_above)
    bins_below = torch.gather(bins, -1, inds_below)
    bins_above = torch.gather(bins, -1, inds_above)

    denom = (cdf_above - cdf_below)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)

    return samples


def draw_nerf(pres, values, depths, sharpen=False):
    """
    Do the NeRF integration based on the given samples.
    Args:
        pres: Densities of the samples, shape [..., s]
        values: Color value of each samples, shape [..., s, d]
        depths: Depth of each sample, shape [..., s]

    Returns:
        Batch dimension is optional.
        image: The expected colors of each ray/pixel, [..., d]
        expected_depth: The expected depth of each ray/pixel, [...,]
        depth_dist: Categorical distribution over the samples, [..., s,]
    """
    num_samples, dim = values.shape[-2:]

    if sharpen:
        pres = pres ** 2

    if pres.isnan().any():
        print('pres nan')

    if values.isnan().any():
        print('values nan')

    segment_sizes = depths[..., 1:] - depths[..., :-1]

    # Log prob that the segment between samples i and i+1 is empty
    # Attributing the density of sample i to that segment.
    log_prob_segment_empty = -pres[..., :-1] * segment_sizes
    # Log prob that Everything up until segment i+1 is empty
    log_prob_ray_empty = log_prob_segment_empty.cumsum(-1)

    # Prepend 0 to get the log prob that everything until segment i is empty
    log_prob_ray_empty_shifted = torch.cat((torch.zeros_like(log_prob_ray_empty[..., :1]),
                                            log_prob_ray_empty), -1)

    prob_ray_empty_shifted = torch.exp(log_prob_ray_empty_shifted)

    # Surface density at the point of sample i
    surface_density = pres * prob_ray_empty_shifted
    total_density = surface_density.sum(-1)
    bg_prob = torch.exp(log_prob_ray_empty[..., -1])
    alpha = 1. - bg_prob

    depth_dist = surface_density / total_density.unsqueeze(-1)

    expected_values = (values * depth_dist.unsqueeze(-1)).sum(-2)

    expected_depth = (depth_dist * depths).sum(-1)

    image = torch.cat((expected_values, alpha.unsqueeze(-1)), -1)

    return image, expected_depth, depth_dist


def draw_nerf_train(pres, values, depths, sharpen=False):
    """
    Do the NeRF integration based on the given samples.
    This is a direct port from the original NeRF code. It is more numerically stable for training
    than the method above.
    Args:
        Batch dimension is optional.
        pres: Densities of the samples, shape [..., s]
        values: Color value of each samples, shape [..., s, d]
        depths: Depth of each sample, shape [..., s]

    Returns:
        Batch dimension is optional.
        image: The expected colors of each ray/pixel, [..., d]
        expected_depth: The expected depth of each ray/pixel, [...]
        depth_dist: Categorical distribution over the samples, [..., s]
    """
    num_samples, dims = values.shape[-2:]

    if sharpen:
        pres = pres ** 2

    if pres.isnan().any():
        print(pres)
        print('pres nan')

    if values.isnan().any():
        print('values nan')

    segment_sizes = depths[..., 1:] - depths[..., :-1]

    last_segment = torch.ones_like(segment_sizes[..., -1:]) * 1e10
    segment_sizes_ext = torch.cat((segment_sizes, last_segment), -1)
    # Log prob that the segment between samples i and i+1 is empty
    # Attributing the density of sample i to that segment.
    prob_segment_empty = torch.exp(-pres * segment_sizes_ext)
    alpha = 1. - prob_segment_empty
    # Log prob that Everything up until segment i+1 is empty
    prob_ray_empty = (prob_segment_empty + 1e-10).cumprod(-1)

    # Prepend 0 to get the log prob that everything until segment i is empty
    prob_ray_empty_shifted = torch.cat((torch.ones_like(prob_ray_empty[..., :1]),
                                        prob_ray_empty[..., :-1]), -1)

    if torch.isnan(alpha).any():
        print('alpha nan')
    segment_probs = alpha * prob_ray_empty_shifted

    total_prob = segment_probs.sum(-1)
    if torch.isnan(total_prob).any():
        print('total density nan')
    bg_prob = prob_ray_empty[..., -1]
    total_alpha = 1. - bg_prob

    expected_values = (values * segment_probs.unsqueeze(-1)).sum(-2)
    expected_depth = (segment_probs * depths).sum(-1)

    image = torch.cat((expected_values, total_alpha.unsqueeze(-1)), -1)

    return image, expected_depth, segment_probs


def zs_to_depths(zs, rays, camera_pos):
    view_axis = -camera_pos
    view_axis = view_axis / np.linalg.norm(view_axis, axis=-1, keepdims=True)
    factors = np.einsum('...i,i->...', rays, view_axis)
    depths = zs / factors
    return depths


def depths_to_world_coords(depths, rays, camera_pos, depth_noise=None, noise_ratio=1.):
    #height, width = depths.shape
    #sensor_width = (0.032 / 320) * width
    #rays = get_camera_rays(camera_pos)
    # TODO: Put this code in a place that makes sense
    if depth_noise is not None:
        noise_indicator = (np.random.random(depths.shape) <= noise_ratio).astype(np.float32)
        depths = depths + noise_indicator * np.random.random(depths.shape) * depth_noise

    surface_points = camera_pos + rays * np.expand_dims(depths, -1)
    return surface_points.astype(np.float32)


def importance_sample_empty_points(surface_points, depths, camera_pos, cutoff=0.98, p_near=0.5):
    num_points = surface_points.shape[0]
    rays = surface_points - camera_pos

    random_intercepts = np.random.random((num_points, 1)).astype(np.float32)

    near_indicator = np.random.binomial(1, p_near, size=(num_points, 1))
    range_bottom = near_indicator * cutoff
    range_top = cutoff + (near_indicator * (1. - cutoff))

    random_intercepts = range_bottom + (range_top - range_bottom) * random_intercepts

    noise_points = camera_pos + (random_intercepts * rays)
    weights = (cutoff * depths * (1 - near_indicator[..., 0]) * 2 +
               (1 - cutoff) * depths * near_indicator[..., 0] * 2)

    return noise_points.astype(np.float32), weights.astype(np.float32), random_intercepts[..., 0]


def sample_empty_points(surface_points, camera_pos, min_quantile=0., max_quantile=1.):
    num_points = surface_points.shape[0]
    rays = surface_points - camera_pos

    random_intercepts = np.random.random((num_points, 1))
    random_intercepts = min_quantile + (max_quantile - min_quantile) * random_intercepts
    noise_points = camera_pos + (random_intercepts * rays)

    return noise_points.astype(np.float32)


