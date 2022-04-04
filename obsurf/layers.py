import torch
import torch.nn as nn
import numpy as np

from obsurf.common import make_2d_grid

import math


class PositionalEncoding(nn.Module):
    def __init__(self, freqs=8, start_freq=0):
        super().__init__()
        self.freqs = freqs
        self.start_freq = start_freq
        print('Positional Encoding params:', self.freqs, self.start_freq)

    def forward(self, coords, rays=None):
        embed_fns = []
        batch_size, num_points, dim = coords.shape

        multipliers = 2**torch.arange(self.start_freq, self.start_freq + self.freqs).float().to(coords) * math.pi
        #print(multipliers)
        multipliers = multipliers.view(1, 1, 1, multipliers.shape[0])

        scaled_coords = coords.unsqueeze(-1) * multipliers

        sines = torch.sin(scaled_coords).view(batch_size, num_points, dim * self.freqs)
        cosines = torch.cos(scaled_coords).view(batch_size, num_points, dim * self.freqs)

        result = torch.cat((coords, sines, cosines), -1)

        assert(result.shape[-1] == (dim + 2 * dim * self.freqs))

        return result


def build_grid(resolution, with_inverse=True):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    if with_inverse:
        return np.concatenate([grid, 1.0 - grid], axis=-1)
    else:
        return grid


class PixelEncoder2d(nn.Module):
    def __init__(self, channels, resolution):
        super().__init__()
        self.channels = channels
        self.resolution = resolution

        self.grid = build_grid(resolution)
        self.proj = nn.Linear(4, channels)

    def forward(self, x):
        grid = self.proj(torch.tensor(self.grid).to(x))
        return x + grid


class PixelEncoder2dFourier(nn.Module):
    def __init__(self, resolution, freqs=8):
        super().__init__()
        self.freqs = freqs
        self.resolution = resolution
        grid = torch.Tensor(build_grid(resolution, with_inverse=False)[0]).float()

        self.pos_encoding = PositionalEncoding(freqs=freqs, start_freq=0)

        grid_enc = self.pos_encoding(grid)
        grid_enc = grid_enc.permute(2, 0, 1)
        self.register_buffer('grid_enc', grid_enc)

    def forward(self, x):
        grid_enc = self.grid_enc.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return torch.cat((x, grid_enc), 1)


class PixelEncoder3d(nn.Module):
    def __init__(self, encode_rays=False):
        super().__init__()
        self.pos_encoding = PositionalEncoding(freqs=16, start_freq=-5)
        self.encode_rays = encode_rays
        if encode_rays:
            self.ray_encoding = PositionalEncoding(freqs=16, start_freq=-5)

    def forward(self, x, camera_pos, rays):
        batchsize, height, width, dims = rays.shape
        c_pos_enc = self.pos_encoding(camera_pos.unsqueeze(1))
        c_pos_enc = c_pos_enc.view(batchsize, c_pos_enc.shape[-1], 1, 1)
        c_pos_enc = c_pos_enc.repeat(1, 1, height, width)
        if self.encode_rays:
            rays = rays.flatten(1, 2)
            rays = self.ray_encoding(rays)
            rays = rays.view(batchsize, height, width, rays.shape[-1])
        rays = rays.permute((0, 3, 1, 2))
        x = torch.cat((x, rays, c_pos_enc), 1)
        return x


class SlotAttention(nn.Module):
    def __init__(self, num_slots, height, width, dim,
                 iters=3, eps=1e-8, hidden_dim=128, self_attention=False,
                 explicit_bg=False, pos_encoder=None, deterministic=False):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.pos_encoder = pos_encoder
        self.deterministic = deterministic

        if pos_encoder is not None:
            pos_enc_dim = dim + 2 + 2 * 2 * pos_encoder.freqs
            att_dim = pos_enc_dim
            grid = make_2d_grid([-0.5, -0.5], [0.5, 0.5], [height, width]).unsqueeze(0)
            self.grid_enc = self.pos_encoder(grid)
        else:
            att_dim = dim
            self.grid_enc = None


        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.explicit_bg = explicit_bg
        if explicit_bg:
            self.bg_slot = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(att_dim, att_dim)
        self.to_k = nn.Linear(att_dim, att_dim)
        self.to_v = nn.Linear(att_dim, att_dim)

        self.gru = nn.GRUCell(att_dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        if self_attention:
            self.self_att = nn.MultiheadAttention(dim, 4)
        else:
            self.self_att = None

    def forward(self, inputs, num_slots=None, coord_mapper=None, get_atts=False):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        if self.explicit_bg:
            n_s -= 1

        mu = self.slots_mu.expand(b, n_s, -1)

        sigma = nn.functional.softplus(self.slots_sigma)
        sigma = sigma.expand(b, n_s, -1)
        if self.deterministic:
            slots = mu
        else:
            sigma = nn.functional.softplus(self.slots_sigma) + 1e-10
            sigma = sigma.expand(b, n_s, -1)
            slots = torch.normal(mu, sigma)

        if self.explicit_bg:
            slots = torch.cat((self.bg_slot.repeat(b, 1, 1), slots), 1)

        inputs = self.norm_input(inputs)
        if self.grid_enc is not None:
            grid_enc = self.grid_enc.repeat(b, 1, 1).to(inputs)
            inputs = torch.cat((inputs, grid_enc), -1)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            norm_slots = self.norm_slots(slots)

            if self.grid_enc is not None:
                slot_positions = coord_mapper(slots)
                slot_pos_enc = self.pos_encoder(slot_positions)
                norm_slots = torch.cat((norm_slots, slot_pos_enc), -1)

            q = self.to_q(norm_slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            if get_atts:
                return attn, slots_prev
            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, updates.shape[-1]),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)

            if self.self_att is not None:
                slots_t = slots.transpose(0, 1)
                self_att_output, _ = self.self_att(slots_t, slots_t, slots_t)
                slots_t = slots_t + self_att_output
                slots = slots_t.transpose(0, 1)

            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

