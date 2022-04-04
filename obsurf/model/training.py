import os, sys
from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist

from obsurf.common import (
    compute_iou, make_2d_grid, make_3d_grid, compute_adjusted_rand_index
)
from obsurf.utils import visualize as vis, nerf as nerf
from obsurf.training import BaseTrainer
from obsurf.data import subsample_clevr, subsample_clevr_batched
from obsurf.checkpoints import CheckpointIO

import math
from collections import defaultdict


def _cube_cull(points, limit=0.5):
    return (torch.abs(points).max(-1)[0] <= 0.5).float()

def _prob_to_density(prob, power=100):
    return prob * power

def _density_to_prob(density, max_density=10.):
    return density / max_density


class Trainer(BaseTrainer):
    ''' Trainer object for the ObSuRF.

    Args:
        model (nn.Module): ObSuRF model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
    '''

    def __init__(self, model, optimizer, device=None, 
                 out_dir=None, threshold=0.5, eval_sample=False, dims=3, l1=0., dataset=None,
                 grad_clip=None, use_occ_ll=False, no_depth_ll=False, no_value_ll=False, vertical=None,
                 use_pixel_loss=None, no_decode=False,
                 anneal_depth_until=None, anneal_depth_start=0, depth_ll=1.,
                 anneal_l1_until=None, anneal_l1_start=0,
                 anneal_color_until=None, anneal_color_start=0,
                 use_world_coords=False,
                 loss_type='depth', depthloss_factor=0.1, train_coarse_samples=48, train_fine_samples=48):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.out_dir = out_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.dims = dims
        self.l1 = l1
        self.dataset = dataset
        self.grad_clip = grad_clip
        self.use_occ_ll = use_occ_ll
        self.no_depth_ll = no_depth_ll
        self.no_value_ll = no_value_ll
        self.vertical = vertical
        self.use_pixel_loss = use_pixel_loss
        self.no_decode = no_decode
        self.depth_ll = depth_ll
        self.anneal_depth_start = anneal_depth_start
        self.anneal_depth_until = anneal_depth_until
        self.anneal_l1_start = anneal_l1_start
        self.anneal_l1_until = anneal_l1_until
        self.anneal_color_start = anneal_color_start
        self.anneal_color_until = anneal_color_until
        self.use_world_coords = use_world_coords
        self.loss_type = loss_type
        self.depthloss_factor = 0.1
        self.train_coarse_samples = train_coarse_samples
        self.train_fine_samples = train_fine_samples

        self.num_nerf_samples = 224
        self.bad_training_steps = 0
        self.mean_grad = 0.
        
        self.vis_dir = os.path.join(out_dir, 'vis')
        if self.vis_dir is not None and not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

    def subsample(self, x, downsample=2):
        if self.dataset == 'clevr3d' or self.dataset == 'multishapenet':
            return subsample_clevr(x)
        else:
            factor = 2**downsample
            return x[factor//2::factor, factor//2::factor]

    def subsample_batched(self, x, downsample=2):
        if self.dataset == 'clevr3d' or self.dataset == 'multishapenet':
            return subsample_clevr_batched(x)
        else:
            factor = 2**downsample
            return x[:, factor//2::factor, factor//2::factor]

    def get_annealing_factor(self, it, start, end):
        if end is not None:
            cur_factor = float(it - start) / (end - start)
            return max(min(1., cur_factor), 0.)
        else:
            return 1.

    def get_depth_ll_factor(self, it):
        return self.get_annealing_factor(it, self.anneal_depth_start, self.anneal_depth_until) * self.depth_ll

    def get_color_ll_factor(self, it):
        return self.get_annealing_factor(it, self.anneal_color_start, self.anneal_color_until)

    def get_l1_factor(self, it):
        return self.get_annealing_factor(it, self.anneal_l1_start, self.anneal_l1_until) * self.l1

    def train_step(self, data, it):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_terms = self.compute_loss(data, it)
        loss = loss.mean(0)

        loss.backward()
        if self.grad_clip is not None:
            norm = nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if norm > self.grad_clip:
                print(f'Clipping gradient from {norm} to {self.grad_clip}')
            self.mean_grad = 0.9 * self.mean_grad + 0.1 * norm
        else:
            norm = 1.

        max_norm = 200.

        if it < 10000 or norm <= max_norm:
            self.optimizer.step()
            self.bad_training_steps = 0
        else:
            print(f'Norm if {norm} > {max_norm}, skipping step entirely')
            self.bad_training_steps += 1

        if self.bad_training_steps > 10:
            print('Training got stuck. Reloading.')
            checkpoint_io = CheckpointIO(self.out_dir, model=self.model, optimizer=self.optimizer)
            load_dict = checkpoint_io.load('model.pt')

        return loss.item()

    def eval_step(self, data, full_scale=False):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold

        # Compute elbo
        inputs = data.get('inputs').to(device)
        input_rays = data.get('input_rays')
        point_rays = data.get('rays')
        camera_pos = data.get('camera_pos')
        if camera_pos is not None:
            camera_pos = camera_pos.to(device)
            input_rays = input_rays.to(device)
            point_rays = point_rays.to(device)
        batch_size = inputs.shape[0]
        kwargs = {}

        self.optimizer.zero_grad()

        with torch.no_grad():
            # For evaluation, we skip the loss annealing and pretend we're done with training.
            loss, loss_terms = self.compute_loss(data, it=1000000)
        eval_dict = loss_terms.copy()
        eval_dict['loss'] = loss

        # Compute iou
        occ_iou = data.get('points.occ')
        points = data.get('points')
        points_values = data.get('points.values')
        if points is not None:
            points = points.to(device)
            with torch.no_grad():
                reps = self.model(points, inputs, sample=self.eval_sample,
                                  camera_pos=camera_pos,
                                  input_rays=input_rays,
                                  decoder_rays=point_rays)

        if occ_iou is not None:
            occ_iou = occ_iou.to(device)
            occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
            occ_iou_hat_np = (reps.global_presence() >= threshold).cpu().numpy()
            iou = compute_iou(occ_iou_np, occ_iou_hat_np)
            eval_dict['iou'] = torch.Tensor(iou)

        if points_values is not None:
            points_values = points_values.to(device)
            recons = reps.global_values()
            mse = ((recons - points_values) ** 2).mean((1, 2))
            eval_dict['mse'] = mse

        masks = data.get('masks')
        if masks is not None:
            masks = masks.to(device)
            if not full_scale and self.dims == 3:
                masks = self.subsample_batched(masks.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            masks = masks.reshape(batch_size, masks.shape[1], masks.shape[2] * masks.shape[3])
            idxs = None
            if camera_pos is None:
                segmentations = F.softmax(reps.local_presence(), 1)
            else:
                with torch.no_grad():
                    idxs = torch.arange(batch_size)[data['view_idxs'] == 0].to(device)
                    if len(idxs) == 0:
                        return eval_dict

                    masks = masks[idxs]
                    camera_pos = camera_pos.to(device)[idxs]
                    input_rays = data.get('input_rays').to(device)[idxs]

                    pixel_encoding, c = self.model.encode_inputs(inputs[idxs], sample=self.eval_sample,
                                                                 camera_pos=camera_pos, rays=input_rays)

                    if not full_scale:
                        input_rays_render = self.subsample_batched(input_rays, downsample=2)
                    else:
                        input_rays_render = input_rays
                    recons, recon_depths, _, _, segmentations = self.render_nerf_batched(
                        c, camera_pos, input_rays_render, num_samples=self.num_nerf_samples,
                        pixel_features=pixel_encoding, input_camera_pos=camera_pos)

                segmentations = segmentations.view(segmentations.shape[0], segmentations.shape[1],
                                                   segmentations.shape[2] * segmentations.shape[3])

                input_images = inputs.permute(0, 2, 3, 1)[idxs]
                input_images_eval = input_images if full_scale else self.subsample_batched(input_images)
                mse = ((input_images_eval - recons[..., :3])**2).mean((1, 2, 3))
                eval_dict['mse'] = mse

                input_depths = data['input_depths'].to(recon_depths)[idxs]
                input_depths_eval = input_depths if full_scale else self.subsample_batched(input_depths)
                depth_error = (input_depths_eval - recon_depths)**2
                depth_mse = depth_error.mean((1, 2))
                eval_dict['depth_mse'] = depth_mse

                fg_masks = 1. - masks[:, 0].float()
                fg_depth_mse = (depth_error.flatten(1, 2) * fg_masks).sum((1)) / fg_masks.sum((1))
                eval_dict['fg_depth_mse'] = fg_depth_mse

            # Ignore true background for ARI computation, following Slot Attention, IODINE, etc.
            masks_no_bg = masks[:, 1:] 
            eval_dict['ari_fg'] = compute_adjusted_rand_index(masks_no_bg, segmentations)
            eval_dict['ari'] = compute_adjusted_rand_index(masks, segmentations)

        return eval_dict

    def eval_samples(self, coords, c, rays, cull_fn=None, density_fn=None,
                     slots_separately=False, pixel_features=None, camera_pos=None,
                     fine=False):
        """
        Args:
            coords: [batch_size, num_rays, num_samples, 3] or
                    [batch_size, num_slots, num_rays, num_samples] if slots_separately
            c: [batch_size, num_slots, c_dim]
            rays: [batch_size, num_rays, 3]
            pixel_features: optionally [batch_size, w, h, c'_dim]
            camera_pos: optionally camera_pos from which pixel_features were obtained [batch_size, 3]
        Return:
            global_pres: [batch_size, num_rays, num_samples] Density aggregated from all slots
            global_values: [batch_size, num_rays, num_samples, 3] Color value aggregated from all slots
            local_pres: [batch_size, num_slots, num_rays, num_samples, 3] Density per slot
            local_values: [batch_size, num_slots, num_rays, num_samples, 3] Color values per slot
        """
        def _process_densities(pres, coords):
            if cull_fn is not None:
                cull = cull_fn(coords)
                pres = pres * cull
            if density_fn is not None:
                pres = density_fn(pres)
            return pres

        if len(coords.shape) == 4:
            batch_size, num_rays, num_samples = coords.shape[:-1]
            num_slots = c.shape[1]
            coords_flat = coords.flatten(1, 2)
        else:
            batch_size, num_slots, num_rays, num_samples = coords.shape[:-1]
            coords_flat = coords.flatten(2, 3)
        rays_ext = rays.unsqueeze(-2).repeat(1, 1, num_samples, 1)
        rays_flat = rays_ext.flatten(1, 2)

        nerf_reps = self.model.decode(coords_flat, c, rays=rays_flat,
                                      pixel_features=pixel_features,
                                      camera_pos=camera_pos, fine=fine)

        global_pres = nerf_reps.global_presence().view(batch_size, num_rays, num_samples)
        global_values = nerf_reps.global_values().view(batch_size, num_rays, num_samples, 3)
        local_pres = nerf_reps.local_presence().view(batch_size, num_slots, num_rays, num_samples)
        local_values = nerf_reps.local_values().view(batch_size, num_slots, num_rays, num_samples, 3)

        return global_pres, global_values, local_pres, local_values

    def render_nerf(self, c, camera_pos, rays,
                    num_samples=None, num_fine_samples=None, num_coarse_samples=None,
                    min_dist=0.035, max_dist=30.,
                    cull_fn=None, density_fn=None, shade=False, sharpen=False, pixel_features=None,
                    input_camera_pos=None, get_local_images=True, deterministic=True):
        """
        Render NeRF rays.
        Args:
            c: scene encoding [n, num_slots, c_dim]
            camera_pos: camera position [n, r, 3]
            rays: [n, r, 3]
        """
        if num_samples is not None:
            num_fine_samples = num_coarse_samples = num_samples

        num_slots = c.shape[1]
        batch_size, num_rays = rays.shape[:2]
        coarse_depths, coarse_coords = nerf.get_nerf_sample_points(camera_pos, rays,
                                                                   num_samples=num_coarse_samples,
                                                                   min_dist=min_dist, max_dist=max_dist,
                                                                   deterministic=deterministic)

        coarse_global_pres, coarse_global_values, coarse_local_pres, coarse_local_values = \
            self.eval_samples(coarse_coords, c, rays, cull_fn=cull_fn, density_fn=density_fn,
                              pixel_features=pixel_features, camera_pos=input_camera_pos,
                              fine=False)

        if not deterministic:
            coarse_img, _, coarse_depth_dist = nerf.draw_nerf_train(
                coarse_global_pres, coarse_global_values, coarse_depths)
        else:
            coarse_img, _, coarse_depth_dist = nerf.draw_nerf(
                coarse_global_pres, coarse_global_values, coarse_depths)

        fine_depths, fine_coords = nerf.get_fine_nerf_sample_points(
            camera_pos, rays, coarse_depth_dist, coarse_depths,
            min_dist=min_dist, max_dist=max_dist, num_samples=num_fine_samples,
            deterministic=deterministic)

        fine_depths = fine_depths.detach()
        fine_coords = fine_coords.detach()

        fine_global_pres, fine_global_values, fine_local_pres, fine_local_values = self.eval_samples(
            fine_coords, c, rays, cull_fn=cull_fn, density_fn=density_fn,
            pixel_features=pixel_features, camera_pos=input_camera_pos,
            fine=True)

        depths_agg = torch.cat((coarse_depths, fine_depths), -1)
        pres_agg = torch.cat((coarse_global_pres, fine_global_pres), -1)
        values_agg = torch.cat((coarse_global_values, fine_global_values), -2)

        depths_agg, sort_idxs = torch.sort(depths_agg, -1)
        pres_agg = torch.gather(pres_agg, -1, sort_idxs)
        values_agg = torch.gather(values_agg, -2, sort_idxs.unsqueeze(-1).expand_as(values_agg))

        if not deterministic:
            global_img, global_depth, depth_dist = nerf.draw_nerf_train(
                pres_agg, values_agg, depths_agg, sharpen=sharpen)
        else:
            global_img, global_depth, depth_dist = nerf.draw_nerf(
                pres_agg, values_agg, depths_agg, sharpen=sharpen)

        if not get_local_images:
            return global_img, global_depth, coarse_img

        local_fine_depths = []
        local_fine_coords = []

        for i in range(num_slots):
            cur_local_pres = coarse_local_pres[:, i]
            cur_local_values = coarse_local_values[:, i]

            _, _, cur_coarse_depth_dist = nerf.draw_nerf(
                cur_local_pres, cur_local_values, coarse_depths)

            cur_fine_depths, cur_fine_coords = nerf.get_fine_nerf_sample_points(
                camera_pos, rays, cur_coarse_depth_dist, coarse_depths,
                min_dist=min_dist, max_dist=max_dist, num_samples=num_fine_samples,
                deterministic=deterministic)

            local_fine_depths.append(cur_fine_depths)
            local_fine_coords.append(cur_fine_coords)

        local_fine_depths = torch.stack(local_fine_depths, 1)
        local_fine_coords = torch.stack(local_fine_coords, 1)

        local_pres_at_global_depths = torch.cat((coarse_local_pres, fine_local_pres), -1)

        _, _, fine_local_pres, fine_local_values = self.eval_samples(
            local_fine_coords, c, rays, cull_fn=cull_fn, density_fn=density_fn,
            slots_separately=True, pixel_features=pixel_features, camera_pos=input_camera_pos,
            fine=True)

        local_depths_agg = torch.cat((coarse_depths.unsqueeze(1).repeat(1, num_slots, 1, 1),
                                      local_fine_depths), -1)
        local_pres_agg = torch.cat((coarse_local_pres, fine_local_pres), -1)
        local_values_agg = torch.cat((coarse_local_values, fine_local_values), -2)

        local_pres_at_global_depths = torch.gather(local_pres_at_global_depths, -1,
                                                   sort_idxs.unsqueeze(1).expand_as(local_pres_at_global_depths))

        local_depths_agg, sort_idxs_local = torch.sort(local_depths_agg, -1)
        local_pres = torch.gather(local_pres_agg, -1, sort_idxs_local)
        local_values = torch.gather(local_values_agg, -2,
                                    sort_idxs_local.unsqueeze(-1).expand_as(local_values_agg))


        local_img, local_depth, _ = nerf.draw_nerf(local_pres, local_values,
                                                   local_depths_agg, sharpen=sharpen)

        segmentation = (depth_dist.unsqueeze(1) *
                        local_pres_at_global_depths / (local_pres_at_global_depths.sum(1, keepdim=True) + 1e-10)).sum(-1)

        return global_img, global_depth, local_img, local_depth, segmentation


    def render_nerf_batched(self, c, camera_pos, rays, pixel_features=None, input_camera_pos=None,
                            **kwargs):
        '''
        Args:
            c: scene encoding [batch_size, num_slots, c_dim]
            camera_pos: positions to render the rays from [batch_size, 3] or [batch_size, 3]
            rays: [batch_size, h, w, 3]
            pixel_features: optionally [batch_size, h', w', c'_dim]
            input_camera_pos: optionally camera_pos where pixel_features were obtained from [batch_size, 3]
        '''
        batch_size, height, width = rays.shape[0:3]
        num_slots = c.shape[1]
        rays_flat = rays.flatten(1, 2)
        num_rays = height * width

        camera_pos_ext = camera_pos.unsqueeze(1).repeat(1, num_rays, 1)

        global_imgs = []
        global_depths = []
        local_imgs = []
        local_depths = []
        segmentations = []

        num_gpus = torch.cuda.device_count()
        max_rays = 512 // num_gpus

        for i in range(0, num_rays, max_rays):
            global_img, global_depth, local_img, local_depth, segmentation = self.render_nerf(
                c, camera_pos_ext[:, i:i+max_rays], rays_flat[:, i:i+max_rays],
                pixel_features=pixel_features, input_camera_pos=input_camera_pos, **kwargs)
            global_imgs.append(global_img)
            global_depths.append(global_depth)
            local_imgs.append(local_img)
            local_depths.append(local_depth)
            segmentations.append(segmentation)

        global_imgs = torch.cat(global_imgs, 1)
        global_depths = torch.cat(global_depths, 1)
        local_imgs = torch.cat(local_imgs, 2)
        local_depths = torch.cat(local_depths, 2)
        segmentations = torch.cat(segmentations, 2)

        global_imgs = global_imgs.view(batch_size, height, width, 4)
        global_depths = global_depths.view(batch_size, height, width)
        local_imgs = local_imgs.view(batch_size, num_slots, height, width, 4)
        local_depths = local_depths.view(batch_size, num_slots, height, width)
        segmentations = segmentations.view(batch_size, num_slots, height, width)

        return global_imgs, global_depths, local_imgs, local_depths, segmentations

    def visualize(self, data, c=None, mode='val'):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        device = self.device

        inputs = data.get('inputs').to(device)
        batch_size = inputs.shape[0]
        masks = data.get('masks')

        model_kwargs = {}
        camera_pos = data.get('camera_pos')
        if camera_pos is not None:
            camera_pos = camera_pos.to(device)
            model_kwargs['camera_pos'] = camera_pos
            input_rays = data.get('input_rays').to(device)
            if self.use_world_coords:
                model_kwargs['rays'] = data['input_points'].to(device)
            else:
                model_kwargs['rays'] = input_rays

        inputs_np = np.transpose(inputs.cpu().numpy(), (0, 2, 3, 1))
        columns = [('inputs', inputs_np, 'image')]

        if c is None:
            with torch.no_grad():
                pixel_encoding, c = self.model.encode_inputs(inputs, sample=self.eval_sample,
                                                             **model_kwargs)
        if self.dims == 2:
            shape = [64, 64]
            p = make_2d_grid([-0.5, -0.5], [0.5, 0.5], shape).to(inputs)
            p = p.expand(batch_size, *p.size())

            with torch.no_grad():
                reps = self.model.decode(p, c, **model_kwargs)

            global_probs = reps.global_presence().view([batch_size] + shape)
            slot_probs_flat = reps.local_presence()
            num_slots = slot_probs_flat.shape[1]
            slot_probs = slot_probs_flat.view([batch_size, num_slots] + shape)

            global_pres = (global_probs >= self.threshold)
            segmentations = slot_probs.argmax(1)

            global_values = reps.global_values()
            slot_values = reps.local_values()

            v_dim = global_values.shape[-1]

            global_values = global_values.view([batch_size] + shape + [v_dim])
            slot_values = slot_values.view([batch_size, num_slots] + shape + [v_dim])

            recons = vis.make_rgba(global_values)  # , alphas=global_probs)
            columns.append(('recons', recons.cpu().numpy(), 'image'))
            slot_imgs = vis.make_rgba(slot_values, alphas=slot_probs).cpu().numpy()

        else:  # self.dims == 3
            if camera_pos is None:
                # Setup default camera position for ShapeNet etc.
                camera_pos = torch.tensor([[1.5, 1.5, 1.5]])
                min_dist = 3. ** (1. / 2.)
                max_dist = 12. ** (1. / 2.)
                input_rays_small = None
                cull_fn = _cube_cull
                density_fn = _prob_to_density
                camera_pos = camera_pos.repeat(batch_size, 1).to(inputs)
                shade = True
            else:
                if self.dataset == 'dexycb':
                    max_dist = 1.6
                    min_dist = 0.01
                else:
                    max_dist = 40.
                    min_dist = 0.035
                density_fn = None
                cull_fn =  lambda x: (x[..., 2] > -0.1).float()
                input_rays_small = self.subsample_batched(input_rays, downsample=2)
                shade = False

            with torch.no_grad():
                render, depths, slot_values, slot_depths, slot_probs = self.render_nerf_batched(
                    c, camera_pos, input_rays_small, pixel_features=pixel_encoding,
                    input_camera_pos=camera_pos,
                    num_samples=self.num_nerf_samples, max_dist=max_dist,
                    min_dist=min_dist, shade=shade, cull_fn=cull_fn, density_fn=density_fn)

            segmentations = slot_probs.argmax(1)
            global_pres = torch.ones_like(segmentations)
            num_slots = slot_probs.shape[1]
            slot_probs_flat = slot_probs.view(batch_size, num_slots,
                                              slot_probs.shape[2] * slot_probs.shape[3])

            render = vis.make_rgba(render)
            columns.append(('render', render.cpu().numpy(), 'image'))
            depth_imgs = vis.make_rgba(depths.unsqueeze(-1) / max_dist)
            columns.append(('depths', depth_imgs.cpu().numpy(), 'image'))

            camera_pos_rot = nerf.rotate_around_z_axis_torch(camera_pos, -(math.pi / 2))
            rays_rot = nerf.rotate_around_z_axis_torch(input_rays_small, -(math.pi / 2))

            with torch.no_grad():
                render_rot, depths_rot, _, _, _ = self.render_nerf_batched(
                    c, camera_pos_rot, rays=rays_rot, pixel_features=pixel_encoding,
                    input_camera_pos=camera_pos,
                    num_samples=self.num_nerf_samples, shade=shade,
                    min_dist=min_dist, max_dist=max_dist, cull_fn=cull_fn, density_fn=density_fn)

            columns.append(('render_rot', render_rot.cpu().numpy(), 'image'))
            depth_imgs_rot = vis.make_rgba(depths_rot.unsqueeze(-1) / 30.)
            columns.append(('depths_rot', depth_imgs_rot.cpu().numpy(), 'image'))
            slot_imgs = slot_values.cpu().numpy()

            with torch.no_grad():
                input_points = data.get('input_points').to(device)
                input_points_small = self.subsample_batched(input_points, downsample=2)
                vis_height, vis_width = input_points_small.shape[1:3]
                input_points_small = input_points_small.flatten(1, 2)
                recon_reps = self.model.decode(input_points_small, c, rays=input_rays_small.flatten(1, 2),
                                               pixel_features=pixel_encoding, camera_pos=camera_pos)
                recons = recon_reps.global_values().view(batch_size, vis_height, vis_width, 3)
                recon_segs = recon_reps.local_presence().argmax(1)
                recon_segs = recon_segs.view(batch_size, vis_height, vis_width)
                columns.append(('colors', recons.cpu().numpy(), 'image'))
                columns.append(('color seg', recon_segs.cpu().numpy(), 'clustering'))

        slot_text = defaultdict(lambda: "")

        for i in range(num_slots):
            text = [slot_text[(j, i)] for j in range(batch_size)]
            columns.append((f'slot {i}', slot_imgs[:, i], 'image', text))

        output_img_path = os.path.join(self.vis_dir, f'clusters-{mode}')
        seg_colors = global_pres * (segmentations + 1)  # Indicate background as zeros
        columns.append(('pred cluster', seg_colors.cpu().numpy(), 'clustering'))

        if masks is not None:
            masks = masks.to(device)
            if self.dims == 3:
                masks = self.subsample_batched(masks.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            masks_flat = masks.reshape(batch_size, masks.shape[1], masks.shape[2] * masks.shape[3])
            masks_no_bg = masks_flat[:, 1:]
            aris = compute_adjusted_rand_index(masks_no_bg, slot_probs_flat).cpu().numpy()
            row_labels = ['ari={:.2f}'.format(ari) for ari in aris]
            true_colors = masks.argmax(1).cpu().numpy()
            columns.append(('real clusters', true_colors, 'clustering'))
        else:
            row_labels = None

        for c in columns:
            print(c[0], c[1].shape, c[2])

        vis.draw_clustering_grid(columns, output_img_path, row_labels=row_labels)

    def compute_nerf_loss(self, data, c, it, pixel_features=None, camera_pos=None, depthloss=None):
        def rgba_composite_white(rgba):
            rgb = rgba[..., :3]
            alpha = rgba[..., 3:]
            result = torch.ones_like(rgb) * (1. - alpha) + rgb * alpha
            return result

        rays = data['rays'].to(self.device)
        values = data['values'].to(self.device)
        query_camera_pos = data['query_camera_pos'].to(self.device)
        fine_image, depths, coarse_img = self.render_nerf(c, query_camera_pos, rays,
                                                          min_dist=3., max_dist=30.,
                                                          num_coarse_samples=self.train_coarse_samples,
                                                          num_fine_samples=self.train_fine_samples,
                                                          pixel_features=pixel_features,
                                                          input_camera_pos=camera_pos,
                                                          get_local_images=False,
                                                          deterministic=False)

        fine_image = rgba_composite_white(fine_image)
        fine_loss = ((fine_image - values)**2).mean((1, 2))
        loss_terms = {'fine_loss': fine_loss}
        loss = fine_loss

        if coarse_img is not None:
            coarse_image = rgba_composite_white(coarse_img)
            coarse_loss = ((coarse_image - values)**2).mean((1, 2))
            loss = loss + coarse_loss
            loss_terms['coarse_loss'] = coarse_loss

        if depthloss is not None:
            gt_depths = data['depths'].to(self.device)
            depth_mse = (depths - gt_depths)**2
            in_bounds = gt_depths < 30.
            depth_mse = depth_mse * in_bounds.float()
            loss_terms['depthloss'] = depth_mse.mean(1) * depthloss
            loss = loss + loss_terms['depthloss']

        return loss, loss_terms

    def compute_ll(self, data, c, it, pixel_features=None, camera_pos=None):
        occ = data.get('points.occ')
        values = data.get('points.values')
        ll_terms = {}

        if self.model.rep_type == 'nerf3d':
            values = data['values'].to(self.device)
            rays = data['rays'].to(self.device)
            empty_points_weights = data['empty_points_weights'].to(self.device)

            batch_size, num_points = rays.shape[:2]

            surface_points = data['surface_points'].to(self.device)
            empty_points = data['empty_points'].to(self.device)

            surface_reps = self.model.decode(surface_points, c, rays=rays,
                                             pixel_features=pixel_features, camera_pos=camera_pos)
            reps = surface_reps

            value_ll = surface_reps.likelihood(values).sum(-1).mean(1)  # Sum over value dims
            if self.no_value_ll:
                ll = torch.zeros_like(value_ll)
            else:
                ll = value_ll * self.get_color_ll_factor(it)
            ll_terms = {'value_ll': value_ll}

            if not self.no_depth_ll:
                empty_reps = self.model.decode(empty_points, c, rays=rays,
                                               pixel_features=pixel_features, camera_pos=camera_pos)
                surface_density = surface_reps.global_presence()
                empty_density = empty_reps.global_presence()
                empty_log_prob = (-empty_density * empty_points_weights)
                depth_ll = (empty_log_prob + torch.log(surface_density)).mean(1)
                ll = ll + self.get_depth_ll_factor(it) * depth_ll
                ll_terms['depth_ll'] = depth_ll

            if self.use_occ_ll:
                p = data.get('points').to(self.device)
                occ_reps = self.model.decode(p, c, rays=rays,
                                             pixel_features=pixel_features, camera_pos=camera_pos)
                occ_density = occ_reps.global_presence()
                occ_probs = _density_to_prob(occ_density)
                occ_ll = dist.Bernoulli(probs=occ_probs).log_prob(occ.to(self.device)).mean(1)
                ll_terms['occ_ll'] = occ_ll
                ll = ll + occ_ll
                slot_probs = occ_reps.local_presence()
            else:
                slot_probs = reps.local_presence()
        else:
            p = data.get('points').to(self.device)
            reps = self.model.decode(p, c)
            if occ is not None:
                ll = reps.likelihood(occ.to(self.device))
                assert(ll.shape[-1] == 1)
                ll = ll[..., 0].mean(1)
            else:
                ll = reps.likelihood(values.to(self.device)).sum(-1).mean(1)  # Sum over value channels
            slot_probs = reps.local_presence()

        if self.l1 > 0.:
            overlap = slot_probs.sum(1) - slot_probs.max(1)[0]

            l1_penalty = (overlap * self.get_l1_factor(it)).mean(1)  # Mean over points
            ll = ll - l1_penalty
            ll_terms['l1'] = overlap.mean(1)

        return ll, ll_terms

    def compute_loss(self, data, it):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        Return:
            loss: The total loss for each example (shape: [batch])
            loss_terms: Dictionary of loss components, each of shape [batch]
        '''
        device = self.device

        inputs = data.get('inputs').to(device)
        camera_pos = data.get('camera_pos')

        enc_kwargs = {}

        if camera_pos is not None:
            camera_pos = camera_pos.to(device)
            enc_kwargs['camera_pos'] = camera_pos
            if self.use_world_coords:
                enc_kwargs['rays'] = data.get('input_points').to(device)
            else:
                enc_kwargs['rays'] = data.get('input_rays').to(device)

        if self.model.probabilistic_c:
            pixel_encoding, q_c = self.model.encode_inputs(inputs, get_dist=True, **enc_kwargs)
            c = q_c.rsample()
        else:
            pixel_encoding, c = self.model.encode_inputs(inputs, **enc_kwargs)

        loss_terms = dict()
        loss = torch.zeros((inputs.shape[0],)).to(device)

        if self.use_pixel_loss is not None:
            pixel_coords = data['input_points'].to(device)
            pixel_coords_pred = self.model.encoder.coord_pred(pixel_encoding)
            pixel_loss = self.use_pixel_loss * ((pixel_coords - pixel_coords_pred)**2).sum(-1).mean((1, 2))
            loss_terms['pixel_loss'] = pixel_loss
            loss = loss + pixel_loss

        if self.no_decode:
            return loss, loss_terms

        if self.model.probabilistic_c:
            kl_c = dist.kl_divergence(q_c, self.model.p0_c).sum((1, 2))
            beta = min(1., it / 20000.)
            loss = loss + beta * kl_c
            loss_terms['kl_c'] = kl_c

        if self.loss_type == 'depth':
            ll, extra_terms = self.compute_ll(data, c, it,
                                           pixel_features=pixel_encoding, camera_pos=camera_pos)
            loss = loss - ll
            loss_terms['likelihood'] = -ll
        elif self.loss_type == 'nerf':
            nerf_loss, extra_terms = self.compute_nerf_loss(data, c, it, pixel_features=pixel_encoding,
                                                  camera_pos=camera_pos)
            loss = loss + nerf_loss
        elif self.loss_type == 'depthnerf':
            nerf_loss, extra_terms = self.compute_nerf_loss(data, c, it, pixel_features=pixel_encoding,
                                                  camera_pos=camera_pos, depthloss=self.depthloss_factor)
            loss = loss + nerf_loss
        else:
            raise ValueError('Unknown loss type', self.loss_type)

        loss_terms.update(extra_terms)

        if torch.isnan(loss).any():
            print([(k, torch.isnan(v).any()) for (k, v) in loss_terms.items()])

        return loss, loss_terms

