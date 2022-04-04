import math

import torch
import numpy as np


def get_base_radius(rays):
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(np.sum((rays[:, :-1, :, :] - rays[:, 1:, :, :])**2, -1))
    # Copy last column
    dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = dx[..., None] * 2 / np.sqrt(12)
    return radii


def get_base_radius_torch(rays):
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = torch.sqrt(torch.sum((rays[:-1, :, :] - rays[1:, :, :])**2, -1))
    # Copy last column
    dx = torch.cat([dx, dx[-2:-1, :]], 0)

    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = dx * 2 / math.sqrt(12)
    return radii


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d * t_mean[..., None]

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1]).to(d)
        null_outer = eye - d[..., :, None] * d[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag=False, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Assumes `d` is normalized.
    Args:
        d: torch.float32 3-vector, the normalized axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
            the paper (setting this to False will cause catastrophic failure).
    Returns:
        a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def expected_sin(x, x_var):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * torch.sin(x)
    y_var = torch.maximum(torch.zeros_like(x), 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2)
    return y, y_var


def integrated_pos_enc(x_coord, min_deg=0, max_deg=16, diag=False):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].
    Args:
        x_coord: a tuple containing: x, torch.ndarray, variables to be encoded. Should
            be in [-pi, pi]. x_cov, torch.ndarray, covariance matrices for `x`.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diag: bool, if true, expects input covariances to be diagonal (full
            otherwise).
    Returns:
        encoded: torch.ndarray, encoded variables.
    """
    if diag:
        x, x_cov_diag = x_coord
        scales = torch.tensor([2**i for i in range(min_deg, max_deg)])
        shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x[..., None, :] * scales[:, None], shape)
        y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)
    else:
        x, x_cov = x_coord
        num_dims = x.shape[-1]
        basis = torch.cat(
                [2**i * torch.eye(num_dims) for i in range(min_deg, max_deg)], 1).to(x)
        y = torch.matmul(x, basis)
        # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
        # to jax.vmap(torch.diag)((basis.T @ covs) @ basis).
        y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)

    return expected_sin(
            torch.cat([y, y + 0.5 * math.pi], axis=-1),
            torch.cat([y_var] * 2, axis=-1))[0]

