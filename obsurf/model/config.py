import torch
import torch.distributions as dist
from torch import nn
import numpy as np

import os

from obsurf.model import models, training
from obsurf import data
from obsurf import config
from obsurf import encoder


encoder_dict = {
    'simple_conv': encoder.ConvEncoder,
    'slot_conv': encoder.ConvSlotEncoder,
}


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the ObSuRF model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder_type = cfg['model']['decoder']
    encoder_type = cfg['model']['encoder']
    dim = cfg['data']['dim']
    rep_type = cfg['model']['rep_type']
    c_dim = cfg['model']['c_dim']
    h_dim = cfg['model']['h_dim'] if 'h_dim' in cfg['model'] else c_dim
    if rep_type == 'occupancy' or rep_type == 'occupancy3d':
        o_dim = 1
    else:
        o_dim = cfg['model']['value_dim'] + 1

    probabilistic_c = 'probabilistic_c' in cfg['model'] and cfg['model']['probabilistic_c']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    if 'coarsefine' in cfg['model'] and cfg['model']['coarsefine']:
        fine_decoder = models.decoder_dict[decoder_type](
            dim=dim, c_dim=c_dim, o_dim=o_dim, rep_type=rep_type,
            **decoder_kwargs)
    else:
        fine_decoder = None

    decoder = models.decoder_dict[decoder_type](
        dim=dim, c_dim=c_dim, o_dim=o_dim, rep_type=rep_type,
        **decoder_kwargs)

    coord_transform = 'coord_transform' in decoder_kwargs and decoder_kwargs['coord_transform']
    if coord_transform:
        encoder_kwargs['pos_encoder'] = decoder.net.pos_encoder
        encoder_kwargs['coord_mapper'] = lambda c: decoder.coord_transform.get_transform(c[..., :c_dim])[1]

    if encoder_type == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim=c_dim)
    elif encoder_type is not None:
        if probabilistic_c:
            c_dim_params = c_dim * 2
        else:
            c_dim_params = c_dim

        encoder = encoder_dict[encoder_type](
            c_dim=c_dim_params,
            h_dim=h_dim,
            **encoder_kwargs
        )
    else:
        encoder = None

    p0_c = get_prior(c_dim, device)
    model = models.Obsurf(
        decoder, encoder=encoder,
        p0_c=p0_c, probabilistic_c=probabilistic_c, device=device,
        fine_decoder=fine_decoder)

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the ObSuRF model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    l1 = cfg['training']['l1'] if 'l1' in cfg['training'] else 0.
    grad_clip = cfg['training']['grad_clip'] if 'grad_clip' in cfg['training'] else None
    use_occ_ll = 'use_occ_ll' in cfg['training'] and cfg['training']['use_occ_ll']
    no_depth_ll = 'no_depth_ll' in cfg['training'] and cfg['training']['no_depth_ll']
    if 'trainer_kwargs' in cfg['training']:
        trainer_kwargs = cfg['training']['trainer_kwargs']
    else:
        trainer_kwargs = dict()

    if 'vertical' in cfg['data']:
        vertical = np.array(cfg['data']['vertical'])
    else:
        vertical = np.array([0., 0., 1.])

    trainer = training.Trainer(
        model, optimizer, device=device,
        out_dir=out_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        dims=cfg['data']['dim'],
        grad_clip=grad_clip,
        use_occ_ll=use_occ_ll,
        no_depth_ll=no_depth_ll,
        vertical=vertical,
        l1=l1,
        dataset=cfg['data']['dataset'],
        **trainer_kwargs
    )

    return trainer


def get_prior(dim, device, **kwargs):
    ''' Returns prior distribution for latent code

    Args:
        dim: Number of dims of the latent code
        device (device): pytorch device
    '''
    p = dist.Normal(
        torch.zeros(dim, device=device),
        torch.ones(dim, device=device)
    )

    return p


