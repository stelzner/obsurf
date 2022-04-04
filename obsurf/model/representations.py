import torch
import torch.nn.functional as F
from torch import distributions as dist

from obsurf import config


def make_representations(x, coords, rep_type, aggregation=None, scale=None):
    if rep_type == 'mixture':
        return MixtureSceneRepresentation(x)
    elif rep_type == 'nerf2d':
        return Nerf2dSceneRepresentation(x, coords, scale=scale)
    elif rep_type == 'occupancy':
        return Occupancy2dSceneRepresentation(x, aggregation)
    elif rep_type == 'occupancy3d':
        return Occupancy3dSceneRepresentation(x, aggregation)
    elif rep_type == 'nerf3d':
        return Nerf3dSceneRepresentation(x, coords)


# These Objects all receive the raw output tensor from the decoder network (ranged [-inf, inf]).
# They provide the following API:
    # - local_values(): Values assigned to each point by the individual segments
    # - local_presence(): Indicator of how likely it is that each segment is present at each point
    # - global_values(): The value assigned by the system as a whole
    # - global_presence(): Indicator of how likely it is that anything is present at each point
    # - likelihood(data): Log likelihood of data given the representation

class MixtureSceneRepresentation():
    def __init__(self, x):
        self.logits = x[..., 0]
        self.values = x[..., 1:]

        self.weights = F.softmax(self.logits, 1)
        self.exp_values = (self.weights.unsqueeze(-1) * self.values).sum(1)

    def local_values(self):
        return self.values

    def local_presence(self):
        return self.weights

    def global_values(self):
        return self.exp_values

    def global_presence(self):
        return torch.ones_like(self.weights[:, 0])

    def likelihood(self, data):
        p_data = dist.Normal(self.exp_values, torch.ones_like(self.exp_values))
        return p_data.log_prob(data)


eps = 1e-10
class Nerf3dSceneRepresentation():
    def __init__(self, x, coords):
        if 'max_density' in config.CFG['model']:
            max_density = config.CFG['model']['max_density']
        else:
            max_density = 10.
        self.mixture = 'realmixture' in config.CFG['model'] and config.CFG['model']['realmixture']
        self.stddev = config.CFG['training']['color_std'] if 'color_std' in config.CFG['training'] else 0.2

        self.rates = torch.sigmoid(x[..., 0])
        self.rates = self.rates * max_density + eps
        self.values = torch.sigmoid(x[..., 1:])

        kernel_type = config.CFG['model']['obj_kernel']

        if kernel_type == 'l2':
            overshoot = compute_overshoot(coords)
            overshoot[:, 0] = 0.
            self.overshoot_penalty = self.rates * overshoot

            self.rates = self.rates * (1. - overshoot)

        #self.rates = torch.clamp(self.rates, eps, 100.)

        self.total_rates = self.rates.sum(1)
        #self.total_rates = torch.clamp(self.total_rates, eps, 100.)

        self.rates_norm = self.rates / self.rates.sum(1).unsqueeze(1)
        self.exp_values = (self.values * self.rates_norm.unsqueeze(-1)).sum(1)

    def local_values(self):
        return self.values

    def local_presence(self):
        return self.rates

    def global_values(self):
        return self.exp_values

    def global_presence(self):
        return self.total_rates

    def likelihood(self, data):
        #print('dev', dev.min().item(), dev.mean().item(), dev.max().item())
        if self.mixture:
            p_data = dist.Normal(self.values, torch.ones_like(self.values) * self.stddev)
            slot_log_probs = p_data.log_prob(data.unsqueeze(1)).sum(-1)
            log_prob = torch.logsumexp(slot_log_probs + torch.log(self.rates_norm), 1)
        else:
            p_data = dist.Normal(self.exp_values, torch.ones_like(self.exp_values) * self.stddev)
            log_prob = p_data.log_prob(data)

        return log_prob


def compute_overshoot(coords):
    # Compute distance from [-0.5, 0.5] unit square/cube.
    overshoot = torch.clamp(torch.abs(coords) - 0.5, min=0.)
    #overshoot = overshoot * scale[:, 1:].unsqueeze(2)
    overshoot = torch.norm(overshoot, dim=-1)

    max_overshoot = 0.5

    overshoot = torch.clamp(overshoot / max_overshoot, 0., 1.)
    return overshoot


class Nerf2dSceneRepresentation():
    def __init__(self, x, coords, scale=None):
        #self.fg_rates = F.softplus(x[:, 1:, :, 0])
        self.fg_rates = torch.sigmoid(x[:, 1:, :, 0]) * 10.
        fg_coords = coords[:, 1:]

        kernel_type = config.CFG['model']['obj_kernel']
        if kernel_type == 'l2':
            overshoot = (fg_coords ** 2).sum(-1)
            overshoot = 1. - torch.exp(-overshoot)
        elif kernel_type == 'bbox':
            overshoot = compute_overshoot(fg_coords)
        else:
            overshoot = None
        #overshoot_penalty = torch.exp(-overshoot * 2)

        #self.overshoot_penalty = torch.clamp(self.fg_rates - 0.01, min=0.) * overshoot_penalty

        if overshoot is not None:
            self.overshoot_penalty = self.fg_rates * overshoot
            self.fg_rates = self.fg_rates * (1. - overshoot)

        self.total_fg_rates = self.fg_rates.sum(1)
        self.values = torch.sigmoid(x[..., 1:])
        self.fg_values = self.values[:, 1:]
        self.bg_values = self.values[:, 0]

        self.p_bg = torch.exp(-self.total_fg_rates)
        self.p_fg = 1. - self.p_bg

        self.p_segments = (self.fg_rates / (self.total_fg_rates.unsqueeze(1) + 1e-10))
        self.exp_fg_values = (self.fg_values * self.p_segments.unsqueeze(-1)).sum(1)
        self.exp_values = (self.exp_fg_values * self.p_fg.unsqueeze(-1) +
                           self.bg_values * self.p_bg.unsqueeze(-1))

    def local_values(self):
        return self.values

    def local_presence(self):
        return torch.cat((torch.ones_like(self.fg_rates[:, :1]), self.fg_rates), 1)
        """
        slot_probs = torch.cat((self.p_bg.unsqueeze(1), self.p_segments * self.p_fg.unsqueeze(1)), 1)
        total_probs = slot_probs.sum(1)
        if torch.max(torch.abs(total_probs - 1.)) > 0.0001:
            print('weird deviation')
            print(total_probs.min().item(), total_probs.max().item(), total_probs.mean().item())
        return slot_probs
        """

    def global_values(self):
        return self.exp_values

    def global_presence(self):
        return torch.ones_like(self.p_fg)

    def likelihood(self, data):
        p_data = dist.Normal(self.exp_values, torch.ones_like(self.exp_values) * 0.1)
        return p_data.log_prob(data)


class Occupancy2dSceneRepresentation():
    def __init__(self, x, aggregator=None):
        assert(x.shape[-1] == 1)
        self.slot_logits = x
        self.slot_dists = dist.Bernoulli(logits=self.slot_logits)

        if aggregator is not None:
            self.global_logits = aggregator(self.slot_logits)
        else:
            assert(self.slot_logits.shape[1] == 1)
            self.global_logits = self.slot_logits.squeeze(1)

        self.global_dists = dist.Bernoulli(logits=self.global_logits)

    def local_values(self):
        return torch.ones_like(self.slot_logits)

    def local_presence(self):
        return self.slot_dists.probs.squeeze(-1)

    def global_values(self):
        return torch.ones_like(self.global_logits)

    def global_presence(self):
        return self.global_dists.probs.squeeze(-1)

    def likelihood(self, data):
        return self.global_dists.log_prob(data)


class Occupancy3dSceneRepresentation():
    def __init__(self, x, coords, aggregator=None):
        assert(x.shape[-1] == 1)
        self.slot_logits = x
        self.slot_dists = dist.Bernoulli(logits=self.slot_logits.squeeze(-1))

        if aggregator is not None:
            self.global_logits = aggregator(self.slot_logits)
        else:
            assert(self.slot_logits.shape[1] == 1)
            self.global_logits = self.slot_logits.squeeze(1)

        self.global_dists = dist.Bernoulli(logits=self.global_logits.squeeze(-1))

    def local_values(self):
        return torch.ones_like(self.slot_logits).repeat(1, 1, 1, 3)

    def local_presence(self):
        return self.slot_dists.probs.squeeze(-1)

    def global_values(self):
        return torch.ones_like(self.global_logits).repeat(1, 1, 3)

    def global_presence(self):
        return self.global_dists.probs.squeeze(-1)

    def likelihood(self, data):
        return self.global_dists.log_prob(data)



