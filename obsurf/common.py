import torch
import numpy as np

from obsurf.utils.visualize import visualize_2d_cluster

import random


def check_nan(x, name):
    isnan = torch.isnan(x)
    if isnan.any():
        print(f'{name} is NaN!')
        return isnan.int().argmax().item()
    return -1

def compute_adjusted_rand_index(true_mask, pred_mask):
    """
    Computes the adjusted rand index (ARI) of a given image segmentation, ignoring the background.
    Implementation following https://github.com/deepmind/multi_object_datasets/blob/master/segmentation_metrics.py#L20
    Args:
        true_mask: Integer Tensor of shape [batch_size, n_true_groups, n_points] containing true
            one-hot coded cluster assignments, with background being indicated by zero vectors.
        pred_mask: Integer Tensor of shape [batch_size, n_pred_groups, n_points] containing predicted
            cluster assignments encoded as categorical probabilities.
    """
    batch_size, n_true_groups, n_points = true_mask.shape
    n_pred_groups = pred_mask.shape[1]

    if n_points <= n_true_groups and n_points <= n_pred_groups:
        raise ValueError(
          "adjusted_rand_index requires n_groups < n_points. We don't handle "
          "the special cases that can occur when you have one cluster "
          "per datapoint.")

    true_group_ids = true_mask.argmax(1)
    pred_group_ids = pred_mask.argmax(1)

    # Convert to one-hot ('oh') representations
    true_mask_oh = true_mask.float()
    pred_mask_oh = torch.eye(n_pred_groups).to(pred_mask)[pred_group_ids].transpose(1, 2)

    n_points_fg = true_mask_oh.sum((1, 2))

    nij = torch.einsum('bip,bjp->bji', pred_mask_oh, true_mask_oh)

    nij = nij.double()  # Cast to double, since the expected_rindex can introduce numerical inaccuracies

    a = nij.sum(1)
    b = nij.sum(2)

    rindex = (nij * (nij - 1)).sum((1, 2))
    aindex = (a * (a - 1)).sum(1)
    bindex = (b * (b - 1)).sum(1)
    expected_rindex = aindex * bindex / (n_points_fg * (n_points_fg - 1))
    max_rindex = (aindex + bindex) / 2

    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    # We can get NaN in case max_rindex == expected_rindex. This happens when both true and
    # predicted segmentations consist of only a single segment. Since we are allowing the
    # true segmentation to contain zeros (i.e. background) which we ignore, it suffices
    # if the foreground pixels belong to a single segment.

    # We check for this case, and instead set the ARI to 1.

    def _fg_all_equal(values, bg):
        """
        Check if all pixels in values that do not belong to the background (bg is False) have the same
        segmentation id.
        Args:
            values: Segmentations ids given as integer Tensor of shape [batch_size, n_points]
            bg: Binary tensor indicating background, shape [batch_size, n_points]
        """
        fg_ids = (values + 1) * (1 - bg.int()) # Move fg ids to [1, n], set bg ids to 0
        example_fg_id = fg_ids.max(1, keepdim=True)[0]  # Get the id of an arbitrary fg cluster.
        return torch.logical_or(fg_ids == example_fg_id[..., :1],  # All pixels should match that id...
                                bg  # ...or belong to the background.
                               ).all(-1)

    background = (true_mask.sum(1) == 0)
    both_single_cluster = torch.logical_and(_fg_all_equal(true_group_ids, background),
                                            _fg_all_equal(pred_group_ids, background))

    # Ensure that we are only (close to) getting NaNs in exactly the case described above.
    matching = (both_single_cluster == torch.isclose(max_rindex, expected_rindex))

    if not matching.all().item():
        offending_idx = matching.int().argmin()

    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def make_2d_grid(bb_min, bb_max, shape):
    size = shape[0] * shape[1]
    rows =  np.linspace(bb_min[0], bb_max[0], shape[0])
    cols =  np.linspace(bb_min[1], bb_max[1], shape[1])
    rows, cols = np.meshgrid(rows, cols, indexing='ij')
    points = np.stack((rows, cols), -1)
    return torch.tensor(points).view(size, 2)

def transform_points(points, transform):
    ''' Transforms points with regard to passed camera information.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert(points.size(2) == 3)
    assert(transform.size(1) == 3)
    assert(points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points @ R.transpose(1, 2) + t.transpose(1, 2)
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ K.transpose(1, 2)

    return points_out


def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv


def transform_points_back(points, transform):
    ''' Inverts the transformation.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert(points.size(2) == 3)
    assert(transform.size(1) == 3)
    assert(points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points - t.transpose(1, 2)
        points_out = points_out @ b_inv(R.transpose(1, 2))
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ b_inv(K.transpose(1, 2))

    return points_out


def project_to_camera(points, transform):
    ''' Projects points to the camera plane.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    p_camera = transform_points(points, transform)
    p_camera = p_camera[..., :2] / p_camera[..., 2:]
    return p_camera


def get_camera_args(data, loc_field=None, scale_field=None, device=None):
    ''' Returns dictionary of camera arguments.

    Args:
        data (dict): data dictionary
        loc_field (str): name of location field
        scale_field (str): name of scale field
        device (device): pytorch device
    '''
    Rt = data['inputs.world_mat'].to(device)
    K = data['inputs.camera_mat'].to(device)

    if loc_field is not None:
        loc = data[loc_field].to(device)
    else:
        loc = torch.zeros(K.size(0), 3, device=K.device, dtype=K.dtype)

    if scale_field is not None:
        scale = data[scale_field].to(device)
    else:
        scale = torch.zeros(K.size(0), device=K.device, dtype=K.dtype)

    Rt = fix_Rt_camera(Rt, loc, scale)
    K = fix_K_camera(K, img_size=137.)
    kwargs = {'Rt': Rt, 'K': K}
    return kwargs


def fix_Rt_camera(Rt, loc, scale):
    ''' Fixes Rt camera matrix.

    Args:
        Rt (tensor): Rt camera matrix
        loc (tensor): location
        scale (float): scale
    '''
    # Rt is B x 3 x 4
    # loc is B x 3 and scale is B
    batch_size = Rt.size(0)
    R = Rt[:, :, :3]
    t = Rt[:, :, 3:]

    scale = scale.view(batch_size, 1, 1)
    R_new = R * scale
    t_new = t + R @ loc.unsqueeze(2)

    Rt_new = torch.cat([R_new, t_new], dim=2)

    assert(Rt_new.size() == (batch_size, 3, 4))
    return Rt_new


def fix_K_camera(K, img_size=137):
    """Fix camera projection matrix.

    This changes a camera projection matrix that maps to
    [0, img_size] x [0, img_size] to one that maps to [-1, 1] x [-1, 1].

    Args:
        K (np.ndarray):     Camera projection matrix.
        img_size (float):   Size of image plane K projects to.
    """
    # Unscale and recenter
    scale_mat = torch.tensor([
        [2./img_size, 0, -1],
        [0, 2./img_size, -1],
        [0, 0, 1.],
    ], device=K.device, dtype=K.dtype)
    K_new = scale_mat.view(1, 3, 3) @ K
    return K_new

