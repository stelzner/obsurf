import torch
import torch.nn as nn
from torch import distributions as dist

from obsurf.model import decoder
from obsurf.model.representations import make_representations


# Decoder dictionary
decoder_dict = {
    'mono': decoder.MonoDecoder,
    'slotted': decoder.SlottedDecoder,
}



class Obsurf(nn.Module):
    ''' Obsurf model class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        p0_c (dist): prior distribution for latent code c
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, p0_c=None, probabilistic_c=False, device=None,
                 fine_decoder=None):
        super().__init__()
        if p0_c is None:
            p0_c = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder.to(device)
        self.fine_decoder = fine_decoder
        if self.fine_decoder is not None:
            self.fine_decoder = self.fine_decoder.to(device)

        self.rep_type = self.decoder.rep_type

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self.p0_c = p0_c
        self.probabilistic_c = probabilistic_c

    def forward(self, p, inputs, sample=True, camera_pos=None, input_rays=None, decoder_rays=None):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z and c
        Return:
            Scene representation (model/representations.py)
        '''
        batch_size = p.size(0)
        pixel_encoding, c = self.encode_inputs(inputs, sample=sample, camera_pos=camera_pos, rays=input_rays)
        return self.decode(p, c, rays=decoder_rays)

    def decode(self, p, c, rays=None, pixel_features=None, camera_pos=None, fine=False):
        if fine and self.fine_decoder is not None:
            rep_params, p_transformed = self.fine_decoder(p, c, rays=rays, pixel_features=pixel_features,
                                                     camera_pos=camera_pos)
        else:
            rep_params, p_transformed = self.decoder(p, c, rays=rays, pixel_features=pixel_features,
                                                     camera_pos=camera_pos)
        return make_representations(rep_params, p_transformed, self.rep_type)

    def encode_inputs(self, inputs, sample=False, get_dist=False, **kwargs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        pixel_encoding, c_params = self.encoder(inputs, **kwargs)
        if self.probabilistic_c:
            num_params = c_params.shape[-1]
            c_means = c_params[..., :num_params//2]
            c_log_stdevs = c_params[..., num_params//2:]
            q_c = dist.Normal(c_means, torch.exp(c_log_stdevs))
            if get_dist:
                return pixel_encoding, q_c
            if sample:
                return pixel_encoding, q_c.rsample()
            c = c_means
        else:
            c = c_params

        return pixel_encoding, c

