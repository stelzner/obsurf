import torch
import torch.nn as nn
import torch.nn.functional as F

from obsurf.model.representations import make_representations
from obsurf.model.decoder_modules import (
    NerfDecoder,
    CoordTransform,
)
from obsurf.utils.nerf import project_to_image_plane

import numpy as np

import math


class MonoDecoder(nn.Module):
    def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_size=64,
                 num_layers=3, rep_type='occupancy', out_dim=1,
                 decoder_arch='cbatchnorm', **kwargs):
        super().__init__()
        self.rep_type = rep_type
        if decoder_arch == 'nerf':
            self.net = NerfDecoder(dim=dim, z_dim=z_dim, c_dim=c_dim, o_dim=out_dim,
                                   num_layers=num_layers, hidden_size=hidden_size,
                                   **kwargs)
        else:
            raise ValueError('Unknown decoder architecture:', decoder_arch)

    def forward(self, p, z, c, **kwargs):
        output = self.net(p, z, c).unsqueeze(1)

        return make_representations(output, self.rep_type)


class SlottedDecoder(nn.Module):
    def __init__(self, dim=3, c_dim=128, o_dim=1, hidden_size=64,
                 num_layers=3, aggregation=None, decoder_arch='cbatchnorm',
                 rep_type='occlusion', coord_transform=False,
                 coord_transform_map=False, bae_slots=None, 
                 use_pixel_features=False, pixel_feature_dropout=None,
                 color_only_pixel_features=False, **kwargs):
        super().__init__()
        self.bae_slots = bae_slots
        if bae_slots is not None:
            o_dim *= bae_slots
        self.rep_type = rep_type
        self.o_dim = o_dim
        self.use_pixel_features = use_pixel_features
        self.pixel_feature_dropout = pixel_feature_dropout
        self.color_only_pixel_features = color_only_pixel_features

        self.coord_transform = CoordTransform(dim, c_dim=c_dim,
                                              coord_transform=coord_transform,
                                              coord_transform_map=coord_transform_map,
                                              skip_bg=rep_type=='nerf2d')
        c_dim -= self.coord_transform.used_cs()

        if pixel_feature_dropout is not None:
            self.dropout = nn.Dropout(p=pixel_feature_dropout)

        num_conditioning_features = c_dim * 2 if use_pixel_features and not color_only_pixel_features else c_dim

        num_color_features = c_dim if color_only_pixel_features else 0

        if decoder_arch == 'cbatchnorm':
            self.net = DecoderCBatchNorm(dim=dim, c_dim=num_conditioning_features,
                                         o_dim=o_dim, num_layers=num_layers,
                                         hidden_size=hidden_size)
        elif decoder_arch == 'nerf':
            self.net = NerfDecoder(dim=dim, c_dim=num_conditioning_features, o_dim=o_dim,
                                   num_layers=num_layers, hidden_size=hidden_size,
                                   num_color_features=num_color_features,
                                   **kwargs)
        else:
            raise ValueError('Unknown decoder architecture:', decoder_arch)
        self.aggregator = Aggregator(aggregation)

    def forward(self, p, c, rays=None, pixel_features=None, camera_pos=None):
        """
        Args:
            p: 2d or 3d coords of points to decode. Either [batch_size, num_points, point_dim]
               or [batch_size, num_slots, num_points, point_dim].
            c: Slots [batch_size, num_slots, slot_dim]
            rays: Optionally ray directions [batch_size, num_points, 3]
            pixel_features: Optionally pixel feature map [batch_size, w, h, feature_dim]
            camera_pos: Optionally coords of camera from which pixel_features were obtained [batch_size, 3]
        """
        _, num_slots, c_dim = c.shape

        if len(p.shape) == 3:
            batch_size, num_points, point_dim = p.shape
            p_ext = p.unsqueeze(1).repeat(1, num_slots, 1, 1)
        else:
            batch_size, num_slots, num_points, point_dim = p.shape
            p_ext = p

        p_transformed, c = self.coord_transform(p_ext, c)

        transform = self.coord_transform.get_transform(c)
        if transform is not None:
            scale = transform[0]
        else:
            scale = None

        c_dim = c.shape[-1]

        c_flat = c.flatten(0, 1)
        c_flat = c_flat.unsqueeze(1).repeat(1, num_points, 1)

        p_transformed_flat = p_transformed.flatten(0, 1)

        if rays is not None:
            rays = rays.unsqueeze(1).repeat(1, num_slots, 1, 1).view(batch_size * num_slots, num_points, 3)

        color_features = None
        if self.use_pixel_features:
            height, width, pixel_feature_dim = pixel_features.shape[1:]
            pixel_features = pixel_features.unsqueeze(1).repeat(1, num_slots, 1, 1, 1).flatten(0, 1)

            camera_pos_ext = camera_pos.unsqueeze(1).repeat(1, num_slots, 1).flatten(0, 1)
            img_p = project_to_image_plane(camera_pos_ext, p_transformed_flat)
            img_x = img_p[..., 0]
            img_y = img_p[..., 1]
            img_x = ((img_x + 0.5) * width).to(torch.int64)
            img_y = ((img_y + 0.5) * height).to(torch.int64)

            img_x = torch.clip(img_x, 0, width-1)
            img_y = torch.clip(img_y, 0, height-1)

            #img_p = ((img_p + 0.5) * torch.tensor([[[width, height]]]).to(img_p)).to(torch.int64)

            batch_idx = torch.arange(pixel_features.shape[0]).unsqueeze(1)
            # [batch_size * num_slots, num_points, feature_dim]
            sel_pixel_features = pixel_features[batch_idx, img_y, img_x]

            if self.pixel_feature_dropout is not None:
                dropout_mask = torch.ones_like(sel_pixel_features[:, :, :1])
                dropout_mask = self.dropout(dropout_mask)
                sel_pixel_features = sel_pixel_features * dropout_mask

            if self.color_only_pixel_features:
                color_features = sel_pixel_features
            else:
                c_flat = torch.cat((c_flat, sel_pixel_features), -1)

        slot_outputs = self.net(p_transformed_flat, c_flat, rays=rays, color_features=color_features)
        if self.bae_slots is not None:
            assert(num_slots == 1)
            slot_outputs  = slot_outputs.view(batch_size, num_points, self.bae_slots,
                                              self.o_dim // self.bae_slots)
            slot_outputs = slot_outputs.permute(0, 2, 1, 3)
            p_transformed = p_transformed_flat.view(batch_size, 1, num_points, point_dim)
            p_transformed = p_transformed.repeat(1, self.bae_slots, 1, 1)
        else:
            slot_outputs = slot_outputs.view(batch_size, num_slots, num_points, self.o_dim)
            p_transformed = p_transformed_flat.view(batch_size, num_slots, num_points, point_dim)
        return slot_outputs, p_transformed
        #return make_representations(slot_outputs, p_transformed, self.rep_type, self.aggregator,
                                    #scale=scale)


class Aggregator(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.eps = 1e-10

    def forward(self, out):
        e_values = None
        if self.method == 'sum':
            out = out.sum(1)
        elif self.method == 'max':
            out, _ = out.max(1)
        elif self.method == 'indep':
            comp_log_p_segments = -torch.logsumexp(torch.stack((out, torch.zeros_like(out)), -1), -1)
            # -torch.logaddexp(out, torch.zeros_like(out))
            comp_log_p = comp_log_p_segments.sum(1)
            out = torch.log(1 - torch.exp(comp_log_p) + self.eps) - comp_log_p
        elif self.method == 'mixture':
            # Computer log variance from logits k
            comp_log_p_segments = -torch.logsumexp(torch.stack((out, torch.zeros_like(out)), -1), -1)  # -log(1 + e^k) = log p(NOT x)
            log_p_segments = out + comp_log_p_segments  # k - log(1 + e^k) = log p(x)
            logvar = log_p_segments + comp_log_p_segments

            mixture_weights = F.softmax(-logvar, 1)

            out = torch.logsumexp(log_p_segments + mixture_weights, 1) # Compute mixture
        else:
            raise ValueError('Unknown aggregation method', self.method)
        return out



