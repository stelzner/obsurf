import os
import math
import random

import torch
from torch.utils import data
import numpy as np
import yaml
import imageio
from PIL import Image
import cv2

from obsurf.common import make_2d_grid
from obsurf.utils.nerf import (get_camera_rays, depths_to_world_coords, importance_sample_empty_points,
    frustum_cull, zs_to_depths)


def subsample_clevr(rays):
    crop = ((29, 221), (64, 256))  # Get center crop.
    rays = rays[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    rays = rays[1::3, 1::3]
    return rays


def subsample_clevr_batched(rays):
    crop = ((29, 221), (64, 256))  # Get center crop.
    rays = rays[:, crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    rays = rays[:, 1::3, 1::3]
    return rays


def extra_points(min_z, max_z, points_per_z, xlim=(-4., 4.), ylim=(-4., 4.), num_objs=11, zthresh=0.):
    num_points = int((max_z - min_z) * points_per_z)

    xs = np.random.uniform(xlim[0], xlim[1], (num_points,))
    ys = np.random.uniform(ylim[0], ylim[1], (num_points,))
    zs = np.random.uniform(min_z, max_z, (num_points,))

    points = np.stack((xs, ys, zs), -1).astype(np.float32)

    objects = np.zeros((num_points, num_objs)).astype(np.float32)
    objects[:, 0] = (zs < zthresh).astype(np.float32)

    return points, objects


class Clevr3dDataset(data.Dataset):
    def __init__(self, path, mode, max_n=6, max_views=None, points_per_item=2048, do_frustum_culling=False,
                 shapenet=False, max_len=None, importance_cutoff=0.5):
        self.path = path
        self.mode = mode
        self.max_n = max_n
        self.points_per_item = points_per_item
        self.do_frustum_culling = do_frustum_culling
        self.shapenet = shapenet
        self.max_len = max_len
        self.importance_cutoff = importance_cutoff

        self.max_num_entities = 5 if shapenet else 11

        if shapenet:
            self.start_idx, self.end_idx = {'train': (0, 80000),
                                            'val': (80000, 80500),
                                            'test': (90000, 100000)}[mode]
        else:
            self.start_idx, self.end_idx = {'train': (0, 70000),
                                            'val': (70000, 70500),
                                            'test': (85000, 100000)}[mode]

        self.metadata = np.load(os.path.join(path, 'metadata.npz'))
        self.metadata = {k: v for k, v in self.metadata.items()}

        num_objs = (self.metadata['shape'][self.start_idx:self.end_idx] > 0).sum(1)
        num_available_views = self.metadata['camera_pos'].shape[1]
        if max_views is None:
            self.num_views = num_available_views
        else:
            assert(max_views <= num_available_views)
            self.num_views = max_views

        self.idxs = np.arange(self.start_idx, self.end_idx)[num_objs <= max_n]

        print(f'Initialized CLEVR {mode} set, {len(self.idxs)} examples')
        print(self.idxs)

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return len(self.idxs) * self.num_views

    def __getitem__(self, idx, noisy=True):
        scene_idx = idx % len(self.idxs)
        view_idx = idx // len(self.idxs)

        scene_idx = self.idxs[scene_idx]

        imgs = [np.asarray(imageio.imread(
            os.path.join(self.path, 'images', f'img_{scene_idx}_{v}.png')))
            for v in range(self.num_views)]
        depths = [np.asarray(imageio.imread(
            os.path.join(self.path, 'depths', f'depths_{scene_idx}_{v}.png')))
            for v in range(self.num_views)]

        imgs = [img[..., :3].astype(np.float32) / 255 for img in imgs]
        # Convert 16 bit integer depths to floating point numbers.
        # 0.025 is the normalization factor used while drawing the depthmaps.
        depths = [d.astype(np.float32) / (65536 * 0.025) for d in depths]

        input_img = np.transpose(imgs[view_idx], (2, 0, 1))

        metadata = {k: v[scene_idx] for (k, v) in self.metadata.items()}

        input_camera_pos = metadata['camera_pos'][view_idx]

        all_rays = []
        all_camera_pos = metadata['camera_pos'][:self.num_views]
        for i in range(self.num_views):
            cur_rays = get_camera_rays(all_camera_pos[i], noisy=False)
            all_rays.append(cur_rays)
        all_rays = np.stack(all_rays, 0)

        if self.shapenet:
            # For the shapenet dataset, the depth images represent the z-coordinate in camera space.
            # Here, we convert this into Euclidian depths.
            new_depths = []
            for i in range(self.num_views):
                new_depth = zs_to_depths(depths[i], all_rays[i], all_camera_pos[i])
                new_depths.append(new_depth)
            depths = np.stack(new_depths, 0)

        example = dict(metadata)
        if self.shapenet:
            # We're not loading the path to the model files into PyTorch, since those are strings.
            del example['shape_file']

        example['view_idxs'] = view_idx
        example['camera_pos'] = input_camera_pos.astype(np.float32)
        example['inputs'] = input_img
        example['input_rays'] = all_rays[view_idx].astype(np.float32)
        if self.mode != 'train':
            example['input_depths'] = depths[view_idx]

        example['input_points'] = depths_to_world_coords(depths[view_idx],
                                                         example['input_rays'],
                                                         example['camera_pos'])

        all_values = np.reshape(np.stack(imgs, 0), (self.num_views * 240 * 320, 3))
        all_depths = np.reshape(np.stack(depths, 0), (self.num_views * 240 * 320,))
        all_rays = np.reshape(all_rays, (self.num_views * 240 * 320, 3))
        all_camera_pos = np.tile(np.expand_dims(all_camera_pos, 1), (1, 240 * 320, 1))
        all_camera_pos = np.reshape(all_camera_pos, (self.num_views * 240 * 320, 3))

        num_points = all_rays.shape[0]

        # If we have fewer points than we want, sample with replacement
        replace = num_points < self.points_per_item
        sampled_idxs = np.random.choice(np.arange(num_points),
                                        size=(self.points_per_item,),
                                        replace=replace)

        rays = all_rays[sampled_idxs]
        camera_pos = all_camera_pos[sampled_idxs]
        values = all_values[sampled_idxs]
        depths = all_depths[sampled_idxs]

        surface_points_base = depths_to_world_coords(depths, rays, camera_pos)

        empty_points, empty_points_weights, empty_t1 = importance_sample_empty_points(
            surface_points_base, depths, camera_pos, cutoff=self.importance_cutoff)


        if noisy:
            depth_noise = 0.07 if noisy else None
            surface_points = depths_to_world_coords(depths, rays, camera_pos, depth_noise=depth_noise)
        else:
            surface_points = surface_points_base

        if self.do_frustum_culling:
            # Cull those points which lie outside the input view
            visible = frustum_cull(surface_points, input_camera_pos, rays)

            surface_points = surface_points[visible]
            empty_points = empty_points[visible]
            values = values[visible]
            depths = depths[visible]
            rays = rays[visible]
            camera_pos = camera_pos[visible]

        example['surface_points'] = surface_points
        example['empty_points'] = empty_points

        example['empty_points_weights'] = empty_points_weights
        example['query_camera_pos'] = camera_pos.astype(np.float32)
        example['values'] = values
        example['rays'] = rays
        example['depths'] = depths

        if self.mode != 'train':
            mask_idx = imageio.imread(os.path.join(self.path, 'masks', f'masks_{scene_idx}_{view_idx}.png'))
            mask = np.zeros((240, 320, self.max_num_entities), dtype=np.uint8)
            np.put_along_axis(mask, np.expand_dims(mask_idx, -1), 1, axis=2)
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
            example['masks'] = mask

        return example


def add_jitter(points, jitter, resolution=64):
    points = points + torch.normal(torch.zeros_like(points),
                                   torch.ones_like(points) / (resolution * jitter))
    return points


class Clevr2dDataset(data.Dataset):
    def __init__(self, path, mode, max_n=6, width=64, height=64, max_len=None,
                 points_subsample=512, jitter=False, shapenet=False):
        self.path = path
        self.width = width
        self.height = height
        self.mode = mode
        self.max_len = max_len
        self.points_subsample = points_subsample
        self.jitter = jitter
        self.shapenet = shapenet

        if shapenet:
            self.start_idx, self.end_idx = {'train': (0, 80000),
                                            'val': (80000, 85000),
                                            'val_unseen': (90000, 100000)}[mode]
        else:
            self.start_idx, self.end_idx = {'train': (0, 70000),
                                            'val': (70000, 75000),
                                            'test': (85000, 100000)}[mode]


        self.metadata = np.load(os.path.join(path, 'metadata.npz'))
        self.metadata = {k: v for k, v in self.metadata.items()}

        num_objs = (self.metadata['shape'][self.start_idx:self.end_idx] > 0).sum(1)

        self.idxs = np.arange(self.start_idx, self.end_idx)[num_objs <= max_n]
        print(f'Initialized CLEVR 2d {mode} set, {len(self.idxs)} examples')
        print(self.idxs)

        self.points = make_2d_grid([-0.5, -0.5], [0.5, 0.5], [self.height, self.width]).float()

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]

        if self.shapenet:
            img_file = f'img_{idx}_0.png'
        else:
            img_file = f'img_{idx}.png'

        img = np.asarray(imageio.imread(os.path.join(self.path, 'images', img_file)))[..., :3]

        img = subsample_clevr(img)  # , resample='bilinear', width=self.width, height=self.height)
        img = img.astype(np.float32) / 255
        img_tr = np.transpose(img, (2, 0, 1))

        if self.mode == 'train' and self.jitter:
            assert(self.width == self.height)  # We're being lazy here.
            points = add_jitter(self.points, self.jitter, resolution=self.width)
        else:
            points = self.points

        example = dict()

        example.update({
                'inputs': img_tr,
                'points': points,
                'points.values': np.reshape(img, (self.height * self.width, 3)),
           })

        if self.mode != 'train':
            if self.shapenet:
                mask_idx = imageio.imread(os.path.join(self.path, 'masks', f'masks_{idx}_0.png'))
                mask = np.zeros((240, 320, 5), dtype=np.uint8)
                np.put_along_axis(mask, np.expand_dims(mask_idx, -1), 1, axis=2)
                mask = subsample_clevr(mask)
                mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
            else:
                mask = np.load(os.path.join(self.path, 'masks', f'mask_{idx}.npz'))['arr_0'].squeeze(-1)
                mask = mask.astype(np.float) / 255.
                mask = np.transpose(subsample_clevr(np.transpose(mask, (1, 2, 0))), (2, 0, 1))
            example['masks'] = mask

        return example



class SpriteDataset(data.Dataset):
    def __init__(self, path, mode, jitter=False, max_len=None):
        self.path = path
        self.data = None
        self.mode = mode
        self.jitter = jitter
        self.max_len = max_len

        with np.load(self.path) as data:
            self.images = data['image'].astype(np.float32) / 255.
            self.images = np.transpose(self.images, (0, 3, 1, 2))
            self.masks = data['mask'].squeeze(-1) // 255
            self.length = self.images.shape[0]
            self.num_channels = self.images.shape[1]

        self.size = 64
        points = make_2d_grid([-0.5, -0.5], [0.5, 0.5], [self.size, self.size])
        self.points = points.float()
        print(f'Loaded {self.length} sprite scenes from {self.path}.')

    def _worker_init_fn(self, *args):
        pass

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        if self.max_len is not None:
            return self.max_len
        return self.length

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        image = self.images[idx]

        values = np.reshape(image, (self.num_channels, self.size * self.size))
        values = np.transpose(values, (1, 0))

        if self.jitter and self.mode == 'train':
            points = add_jitter(self.points, self.jitter, resolution=self.size)
        else:
            points = self.points

        example = {'inputs': image,
                   'points': self.points,
                   'points_iou': self.points,
                   'points.values': values,
                   'masks': self.masks[idx],
                   }

        # Only return occupancy info in the binary case, where the background is always black
        if self.num_channels == 1:
            occ = np.reshape((image > 0.5).astype(np.float32), (self.size * self.size, 1))

            example['points.occ'] = occ

        return example


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

