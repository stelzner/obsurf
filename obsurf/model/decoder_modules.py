import torch
import torch.nn as nn
import torch.nn.functional as F

from obsurf.layers import PositionalEncoding

import math


class CoordTransform(nn.Module):
    def __init__(self, dim, c_dim=None, coord_transform=True, coord_transform_map=False, skip_bg=False):
        super().__init__()
        self.dim = dim
        self.min_scale = 0.05
        if dim == 3:
            self.max_scale = 4.
            self.max_shift = 5.
        else:
            self.max_scale = 1.
            self.max_shift = 0.5

        self.skip_bg = skip_bg
        if coord_transform:
            rot_dims = 1  # We only allow rotation around the z-axis in 3d, so one parameter is enough
            self.transform_size = 2 * dim + rot_dims
            if coord_transform_map:
                self.transform_map = nn.Sequential(
                    nn.Linear(c_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.transform_size),
                )
            else:
                self.transform_map = None
        else:
            self.transform_size = None

    def get_transform(self, c):
        if self.transform_size is None:
            return None
        if self.transform_map is None:
            transform = c[..., :self.transform_size]
        else:
            transform = self.transform_map(c)

        scale = torch.sigmoid(transform[..., :self.dim]) \
                * (self.max_scale - self.min_scale) + self.min_scale
        shift = (torch.sigmoid(transform[..., self.dim:self.dim * 2]) - 0.5) * 2 * self.max_shift
        rotation = transform[..., self.dim * 2:] * math.pi * 2

        if self.skip_bg:
            scale = torch.cat((torch.ones_like(scale[:, :1]), scale[:, 1:]), 1)
            shift = torch.cat((torch.zeros_like(shift[:, :1]), shift[:, 1:]), 1)
            rotation = torch.cat((torch.zeros_like(rotation[:, :1]), rotation[:, 1:]), 1)

        return scale, shift, rotation

    def used_cs(self):
        if self.transform_size is not None and self.transform_map is None:
            return self.transform_size
        else:
            return 0

    def get_rot_matrix(self, rot, inverse=False):
        rot = rot.squeeze(-1)
        cos = torch.cos(rot)
        sin = torch.sin(rot)

        if self.dim == 2:
            result = torch.zeros(list(rot.shape) + [2, 2]).to(rot)
        else:  # self.dim == 3
            # We currently only allow rotation around the z axis.
            result = torch.zeros(list(rot.shape) + [3, 3]).to(rot)
            result[..., 2, 2] = 1.

        result[..., 0, 0] = cos
        result[..., 1, 1] = cos

        if not inverse:
            result[..., 0, 1] = -sin
            result[..., 1, 0] = sin
        else:
            result[..., 0, 1] = sin
            result[..., 1, 0] = -sin

        return result

    def forward(self, x, c, inverse=True):
        transform = self.get_transform(c)
        if transform is None:
            return x, c
        scale, shift, rot = transform
        c = c[..., self.used_cs():]

        # Transformation order is scale -> rotation -> shift
        # If we're inverting, the order is opposite, and the steps are inverted.
        if inverse:
            x = x - shift.unsqueeze(-2)
        else:
            x = x * scale.unsqueeze(-2)

        rot_matrix = self.get_rot_matrix(rot, inverse=inverse)
        rot_matrix = rot_matrix.unsqueeze(-3).expand(list(x.shape) + list(rot_matrix.shape[-1:]))
        x = torch.einsum('...ji,...i->...j', rot_matrix, x)

        if inverse:
            x = x / scale.unsqueeze(-2)
        else:
            x = x + shift.unsqueeze(-2)
        return x, c


class ResUnit(nn.Module):
    def __init__(self, x_dim, c_dim, h_dim):
        super().__init__()
        self.layer1 = nn.Linear(x_dim + c_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, x_dim)

    def forward(self, x, c):
        context = torch.cat((x, c), -1)
        h = F.relu(self.layer1(context))
        return x + self.layer2(h)


class NerfResnet(nn.Module):
    def __init__(self, input_dims, c_dim, o_dim, hidden_size, num_layers=5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dims + c_dim, hidden_size))
        for i in range(num_layers - 2):
            self.layers.append(ResUnit(hidden_size, c_dim, hidden_size))
        self.layers.append(nn.Linear(hidden_size, o_dim))

    def forward(self, x, c):
        c = c.unsqueeze(1).repeat(1, x.shape[1], 1)
        h = self.layers[0](torch.cat((x, c), -1))
        for layer in self.layers[1:-1]:
            h = layer(h, c)
        return self.layers[-1](h)


class NerfNet(nn.Module):
    def __init__(self, input_dims, c_dim, o_dim, hidden_size, num_layers=5):
        super().__init__()
        self.actvn = F.relu
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dims + c_dim, hidden_size))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size + input_dims + c_dim, hidden_size))
        self.layers.append(nn.Linear(hidden_size + input_dims + c_dim, o_dim))

    def forward(self, x, c):
        c = c.unsqueeze(1).repeat(1, x.shape[1], 1)
        conditional = torch.cat((x, c), -1)
        for i, layer in enumerate(self.layers):
            if i == 0:
                h = layer(conditional)
            else:
                h = layer(torch.cat((h, conditional), -1))
            if i < len(self.layers) - 1:
                h = self.actvn(h)

        return h


class NerfAinNet(nn.Module):
    def __init__(self, input_dims, c_dim, o_dim, hidden_size, num_layers=5):
        super().__init__()
        self.actvn = F.relu
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cond_layers = nn.ModuleList()
        self.main_layers = nn.ModuleList()

        self.cond_layers.append(nn.Linear(c_dim, 2 * input_dims))
        for i in range(num_layers - 2):
            self.cond_layers.append(nn.Linear(c_dim, 2 * hidden_size))

        self.main_layers.append(nn.Linear(input_dims, hidden_size))
        for i in range(num_layers - 2):
            self.main_layers.append(nn.Linear(hidden_size, hidden_size))
        self.main_layers.append(nn.Linear(hidden_size, o_dim))

    def forward(self, x, c):
        h = x
        for i in range(self.num_layers - 1):
            cond = self.cond_layers[i](c)
            cond_dims = cond.shape[-1]
            shift = cond[..., :cond_dims // 2]
            scale = cond[..., cond_dims // 2:]
            if len(shift.shape) < len(h.shape):
                shift = shift.unsqueeze(1)
                scale = scale.unsqueeze(1)

            h = (h + shift) * scale

            h = self.actvn(self.main_layers[i](h))
        h = self.main_layers[-1](h)
        return h


class NerfDecoder(nn.Module):
    def __init__(self, dim=3, c_dim=128, o_dim=1, hidden_size=128,
                 num_layers=5, start_freq=0, num_freqs=8, view_dependent_values=False,
                 num_color_features=0, net_arch='mlp'):
        super().__init__()
        self.dim = dim

        self.pos_encoder = PositionalEncoding(freqs=num_freqs, start_freq=start_freq)
        self.actvn = F.relu

        input_dims = dim * 2 * num_freqs
        input_dims += dim
        output_dims = hidden_size + 1 if view_dependent_values else o_dim

        if net_arch == 'resnet':
            self.net = NerfResnet(input_dims, c_dim, output_dims, hidden_size, num_layers=num_layers)
        elif net_arch == 'mlp':
            self.net = NerfNet(input_dims, c_dim, output_dims, hidden_size, num_layers=num_layers)
        elif net_arch == 'ain':
            self.net = NerfAinNet(input_dims, c_dim, output_dims, hidden_size, num_layers=num_layers)
        else:
            raise ValueError('Unknown decoder network architecture:', net_arch)

        if view_dependent_values:
            self.color_predictor = nn.Sequential(
                nn.Linear(dim + hidden_size + num_color_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3))
        else:
            self.color_predictor = None

    def forward(self, x, c=None, color_features=None, **kwargs):
        rays = kwargs['rays']
        pos = self.pos_encoder(x, rays=rays)

        h = self.net(pos, c)

        if self.color_predictor is not None:
            density = h[..., :1]
            h = h[..., 1:]
            if color_features is None:
                color_inputs = torch.cat((h, rays), -1)
            else:
                color_inputs = torch.cat((h, color_features, rays), -1)

            color = self.color_predictor(color_inputs)
            h = torch.cat((density, color), -1)
        return h


