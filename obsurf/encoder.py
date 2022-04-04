import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from obsurf.layers import (SlotAttention, PositionalEncoding, PixelEncoder2d, PixelEncoder3d,
                            PixelEncoder2dFourier)
from obsurf.utils.nerf import project_to_image_plane


class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128, i_dim=1, downsample=0, target_size=7):
        super().__init__()
        self.target_size = target_size
        stride = 2 if downsample > 0 else 1
        self.conv0 = nn.Conv2d(i_dim, 32, 3, stride=stride, padding=1)
        stride = 2 if downsample > 1 else 1
        self.conv1 = nn.Conv2d(32, 64, 3, stride=stride, padding=1)
        stride = 2 if downsample > 2 else 1
        self.conv2 = nn.Conv2d(64, 128, 3, stride=stride, padding=1)
        stride = 2 if downsample > 3 else 1
        self.conv3 = nn.Conv2d(128, 256, 3, stride=stride, padding=1)
        self.fc_out = nn.Linear(256 * target_size**2, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = net.view(batch_size, 256 * self.target_size**2, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


class CustomResnet(models.ResNet):
    def __init__(self, i_dim, block, c_dim=128, h_dim=128, downsample=0, layers=[2, 2, 2, 2]):
        nn.Module.__init__(self)
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = c_dim
        self.downsample = downsample

        self.conv1 = nn.Conv2d(i_dim, c_dim, kernel_size=7, padding=3, stride=self._get_stride(),
                               bias=False)

        self.bn1 = self._norm_layer(c_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, h_dim, layers[0])
        self.layer2 = self._make_layer(block, h_dim, layers[1], stride=self._get_stride())
        self.layer3 = self._make_layer(block, h_dim, layers[2], stride=self._get_stride())
        self.layer4 = self._make_layer(block, c_dim, layers[3], stride=self._get_stride())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_stride(self):
        if self.downsample > 0:
            self.downsample -= 1
            return 2
        return 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def make_custom_resnet(arch, i_dim, c_dim=128, h_dim=128, downsample=0):
    if arch == 'resnet18':
        layers = [2, 2, 2, 2]
        block = models.resnet.BasicBlock
    elif arch == 'resnet34':
        layers = [3, 4, 6, 3]
        block = models.resnet.BasicBlock
    elif arch == 'resnet50':
        layers = [3, 4, 6, 3]
        block = models.resnet.Bottleneck
    else:
        raise ValueError('Unknown resnet architecture')
    return CustomResnet(i_dim, block, c_dim=c_dim, h_dim=h_dim, downsample=downsample, layers=layers)


class ConvSlotEncoder(nn.Module):
    def __init__(self, c_dim=128, i_dim=3, h_dim=128, downsample=0, att_height=None, att_width=None,
                 num_slots=8, num_layers=4, slot_iters=3, self_attention=False, use_camera_pos=False,
                 explicit_bg=False, resnet=False, coord_mapper=None, pos_encoder=None,
                 pred_coords=False, encode_rays=False, mlp_output=False, fourier_pos=False,
                 in_width=64, in_height=64, deterministic=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.att_height = att_height
        self.att_width = att_width
        self.h_dim = h_dim
        self.c_dim = c_dim
        self.use_camera_pos = use_camera_pos
        self.coord_mapper = coord_mapper
        self.explicit_bg = explicit_bg
        self.mlp_output = mlp_output
        self.fourier_pos = fourier_pos

        if use_camera_pos:
            self.pixel_encoder = PixelEncoder3d(encode_rays=encode_rays)
            i_dim += 99  # cpos encoding
            i_dim += 3 + (96 if encode_rays else 0)
        else:
            if fourier_pos:
                self.pixel_encoder = PixelEncoder2dFourier((in_height, in_width))
                i_dim += 8 * 2 * 2 + 2
            else:
                assert(att_height is not None and att_width is not None)
                self.pixel_encoder = PixelEncoder2d(c_dim, (att_height, att_width))

        self.resnet = resnet

        if not resnet:
            for i in range(num_layers):
                stride = 2 if downsample > 0 else 1
                input_dim = i_dim if i == 0 else h_dim
                downsample -= 1
                self.convs.append(nn.Conv2d(input_dim, h_dim, 5, stride=stride, padding=2))
        else:
            print(resnet)
            if isinstance(resnet, str):
                arch = resnet
            else:
                arch = 'resnet18'
            self.resnet = make_custom_resnet(arch, i_dim, c_dim=h_dim, h_dim=h_dim, downsample=downsample)
            # CustomResnet18(i_dim, h_dim, h_dim, downsample=downsample)

        pixel_enc_dim = h_dim if mlp_output else c_dim

        self.norm = nn.LayerNorm(h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, pixel_enc_dim)
        self.actvn = nn.ReLU()

        if pred_coords:
            self.coord_pred = nn.Linear(pixel_enc_dim, 3)
        else:
            self.coord_pred = None

        if mlp_output:
            self.slot_att = nn.Sequential(nn.Flatten(1, 2),
                                          nn.Linear(pixel_enc_dim * att_height * att_width, c_dim * 4),
                                          self.actvn,
                                          nn.Linear(c_dim * 4, c_dim * 2),
                                          self.actvn,
                                          nn.Linear(c_dim * 2, c_dim))
        else:
            self.slot_att = SlotAttention(num_slots, att_height, att_width, c_dim,
                                          iters=slot_iters, hidden_dim=c_dim, self_attention=self_attention,
                                          explicit_bg=explicit_bg, pos_encoder=pos_encoder,
                                          deterministic=deterministic)


    def forward(self, x, camera_pos=None, rays=None):
        batch_size = x.size(0)

        if self.fourier_pos:
            x = self.pixel_encoder(x)
        elif self.use_camera_pos:
            x = self.pixel_encoder(x, camera_pos, rays)

        if not self.resnet:
            for c in self.convs:
                x = self.actvn(c(x))
        else:
            x = self.resnet(x)

        x = x.permute(0, 2, 3, 1)  #  [N, C, H, W] to [N, H, W, C]

        if not self.use_camera_pos and not self.mlp_output and not self.fourier_pos:
            x = self.pixel_encoder(x)

        x = self.norm(x)

        x = self.actvn(self.fc1(x))
        pixel_encoding = self.fc2(x)
        c_params = self.do_slot_att(pixel_encoding)
        return pixel_encoding, c_params

    def do_slot_att(self, pixel_encoding):
        if self.use_camera_pos:
            coord_mapper = lambda x: project_to_image_plane(camera_pos, self.coord_mapper(x))
        else:
            coord_mapper = self.coord_mapper

        pixel_encoding = pixel_encoding.view(pixel_encoding.shape[0],
                                             self.att_height * self.att_width, pixel_encoding.shape[-1])
        slots = self.slot_att(pixel_encoding)
        if self.mlp_output:
            slots = slots.unsqueeze(1)
        return slots

