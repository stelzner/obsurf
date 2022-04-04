import torch
import torch.optim as optim
import numpy as np
import imageio

import os, sys, argparse, math

from obsurf import config
import obsurf.model.config as model_config
from obsurf.checkpoints import CheckpointIO
from obsurf.utils.visualize import visualize_2d_cluster, get_clustering_colors
from obsurf.utils.nerf import rotate_around_z_axis_torch, get_camera_rays
from obsurf.common import make_2d_grid


def downsample(x, steps=2):
    factor = 2**steps
    return x[factor//2::factor, factor//2::factor]

def get_camera_rays_render(camera_pos, **kwargs):
    rays = get_camera_rays(camera_pos, **kwargs)
    if args.downsample is not None:
        return downsample(rays, steps=args.downsample)
    return rays

def lerp(x, y, t):
    return x + (y-x) * t

def easeout(t):
    return -0.5 * t**2 + 1.5 * t

def add_fade(t, t_fade=0.2):
    v_max = 1. / (1. - t_fade)
    acc = v_max / t_fade
    if t <= t_fade:
        return 0.5 * acc * t**2
    pos_past_fade = 0.5 * acc * t_fade**2
    if t <= 1. - t_fade:
        return pos_past_fade + v_max * (t - t_fade)
    else:
        return 1. - 0.5 * acc * (t - 1.)**2


def render2d(model, render_path, c, resolution=(64, 64)):
    height, width = resolution

    max_size = 0.5
    min_size = 0.1

    for frame in range(args.num_frames):
        print(f'{frame}...')
        cur_size = min_size + (max_size - min_size) * (1 - frame / args.num_frames)
        points = make_2d_grid([-cur_size, -cur_size], [cur_size, cur_size], [height, width]).float().to(c)
        points = points + 1 / 128.

        with torch.no_grad():
            reps = model.decoder(points.unsqueeze(0), c)
            values = reps.global_values()
            values = values.view(height, width, 3).cpu().numpy()
            values = (values * 255.).astype(np.uint8)
            imageio.imwrite(os.path.join(render_path, 'renders', f'{frame}.png'), values)


def get_camera_closeup(camera_pos, t, lookup=1.5):
    final_c_pos = 0.3 * camera_pos
    final_c_pos[2] = 0.25 * camera_pos[2]
    orig_track_points = camera_pos
    final_track_point = -lookup * camera_pos / torch.linalg.norm(camera_pos, dim=-1, keepdim=True)
    final_track_point[2] = 0.

    c_dir = (- camera_pos)
    c_dir = c_dir / torch.linalg.norm(c_dir)

    cur_camera_pos = lerp(camera_pos, final_c_pos, t)
    cur_camera_pos[2] = lerp(camera_pos[2], final_c_pos[2], easeout(t))

    t_track_point = cur_camera_pos[2] / -c_dir[2]
    orig_track_point = cur_camera_pos + c_dir * t_track_point

    #cur_track_point = lerp(torch.zeros_like(camera_pos), final_track_point, t)
    cur_track_point = lerp(orig_track_point, final_track_point, t)

    cur_rays = get_camera_rays_render(cur_camera_pos.cpu().numpy(),
                                      c_track_point=cur_track_point.cpu().numpy())
    cur_rays = torch.Tensor(cur_rays).float().cuda()
    return cur_camera_pos, cur_rays


def rotate_camera(camera_pos, rays, t):
    theta = math.pi * 2 * t
    cur_camera_pos = rotate_around_z_axis_torch(camera_pos, theta)
    cur_rays = rotate_around_z_axis_torch(rays, theta)
    return cur_camera_pos, cur_rays

def render3d(trainer, render_path, c, camera_pos, input_rays, motion, pixel_encoding=None):
    cull_fn =  lambda x: (x[..., 2] > -0.1).float()

    seg_colors = get_clustering_colors(8)

    for frame in range(args.num_frames):
        print(f'{frame}...')
        t = frame / args.num_frames
        if args.fade:
            t = add_fade(t)
        if motion == 'rotate':
            cur_camera_pos, cur_rays = rotate_camera(camera_pos, input_rays, t)
        elif motion == 'zoom3d':
            sensor_max = 0.032
            sensor_min = sensor_max / 5
            sensor_cur = lerp(sensor_max, sensor_min, frame / args.num_frames)
            camera_pos_np = camera_pos.cpu().numpy()
            cur_rays = get_camera_rays_render(camera_pos_np, sensor_width=sensor_cur)
            cur_rays = torch.Tensor(cur_rays).float().cuda()
            cur_camera_pos = camera_pos
        elif motion == 'closeup':
            cur_camera_pos, cur_rays = get_camera_closeup(camera_pos, t)
        elif motion == 'rotate_and_closeup':
            t_closeup = ((-math.cos(t * math.pi * 2) + 1) * 0.5) * 0.5
            cur_camera_pos, cur_rays = get_camera_closeup(camera_pos, t_closeup, lookup=1.5)
            cur_camera_pos, cur_rays = rotate_camera(cur_camera_pos, cur_rays, t)
        elif motion == 'graspnet':
            cur_camera_pos = camera_pos
            cur_rays = input_rays

        with torch.no_grad():
            render, depths, local_values, local_depths, seg = trainer.render_nerf_batched(
                c, cur_camera_pos.unsqueeze(0), cur_rays.unsqueeze(0),
                pixel_features=pixel_encoding,
                input_camera_pos=camera_pos.unsqueeze(0),
                num_samples=args.num_samples, max_dist=40., min_dist=0.035, cull_fn=cull_fn,
                sharpen=args.sharpen)
            render = render[0]
            depths = depths[0]
            local_values = local_values[0]
            local_depths = local_depths[0]
            seg = seg[0]
        render = render.cpu().numpy()
        render = (render * 255.).astype(np.uint8)
        depths = depths.cpu().numpy()
        depths = (depths * 0.025 * 65536).astype(np.uint16)
        local_values = local_values.cpu().numpy()
        local_values = (local_values * 255.).astype(np.uint8)
        local_depths = local_depths.cpu().numpy()
        local_depths = (local_depths * 0.025 * 65536).astype(np.uint16)
        seg = seg.argmax(0).cpu().numpy()
        seg_img = visualize_2d_cluster(seg+1, colors=seg_colors)
        seg_img = (seg_img * 255).astype(np.uint8)

        imageio.imwrite(os.path.join(render_path, 'renders', f'{frame}.png'), render)
        imageio.imwrite(os.path.join(render_path, 'depths', f'{frame}.png'), depths)
        imageio.imwrite(os.path.join(render_path, 'segmentations', f'{frame}.png'), seg_img)

        num_slots = local_values.shape[0]
        for slot in range(num_slots):
            imageio.imwrite(os.path.join(render_path, 'slot_renders', f'{slot}-{frame}.png'),
                            local_values[slot])
            imageio.imwrite(os.path.join(render_path, 'slot_depths', f'{slot}-{frame}.png'),
                            local_depths[slot])


def process_scene(sceneid):
    render_path = os.path.join('render', exp_name, str(sceneid), args.name)
    if os.path.exists(render_path):
        if args.force:
            print(f'Path {render_path} exists. Overwriting as requested.')
        else:
            print(f'Path {render_path} exists. Use --force to overwrite.')
            sys.exit()

    os.makedirs(render_path, exist_ok=True)
    subdirs = ['renders', 'depths', 'segmentations', 'slot_renders', 'slot_depths']
    for d in subdirs:
        os.makedirs(os.path.join(render_path, d), exist_ok=True)

    data = val_dataset.__getitem__(sceneid)
    input_image = torch.Tensor(data['inputs']).to(device)
    if 'camera_pos' in data:
        camera_pos = torch.Tensor(data['camera_pos']).to(device)
        input_rays = torch.Tensor(data['input_rays']).to(device)
        input_camera_pos = camera_pos.unsqueeze(0)
        
        encoder_kwargs = {'camera_pos': input_camera_pos, 'rays': input_rays.unsqueeze(0)}

        if args.downsample is not None:
            input_rays = downsample(input_rays, args.downsample)
    else:
        encoder_kwargs = dict()

    input_np = (np.transpose(data['inputs'], (1, 2, 0)) * 255.).astype(np.uint8)
    imageio.imwrite(os.path.join(render_path, 'input.png'), input_np)

    with torch.no_grad():
        pixel_encoding, c = model.encode_inputs(input_image.unsqueeze(0), sample=False, **encoder_kwargs)

    if args.motion in ['rotate', 'zoom3d', 'closeup', 'rotate_and_closeup', 'graspnet']:
        render3d(trainer, render_path, c, camera_pos, input_rays, motion=args.motion, pixel_encoding=pixel_encoding)
    elif args.motion == 'zoom2d':
        render2d(model, render_path, c, resolution=(256, 256))
    else:
        raise ValueError(f'Unknown motion: {args.motion}')


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Render a video of a scene.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--num-frames', type=int, default=360, help='Number of frames to render.')
    parser.add_argument('--num-samples', type=int, default=256, help='Number of samples.')
    parser.add_argument('--sceneid', type=int, default=0, help='Id of the scene to render.')
    parser.add_argument('--sceneid_start', type=int, help='Id of the scene to render.')
    parser.add_argument('--sceneid_stop', type=int, help='Id of the scene to render.')
    parser.add_argument('--downsample', type=int, help='Number of downsampling steps.')
    parser.add_argument('--name', type=str, default='render', help='Name of this sequence.')
    parser.add_argument('--force', action='store_true', help='Overwrite existing sequence.')
    parser.add_argument('--motion', type=str, default='rotate', help='Type of sequence.')
    parser.add_argument('--sharpen', action='store_true', help='Square density values for sharper surfaces.')
    parser.add_argument('--parallel', action='store_true', help='Wrap model in DataParallel.')
    parser.add_argument('--fade', action='store_true', help='Add fade in/out.')
    parser.add_argument('--it', type=int, help='Iteration of the model to load.')


    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    out_dir = cfg['training']['out_dir']
    if out_dir == 'HERE!':
        out_dir = os.path.dirname(args.config)
        cfg['training']['out_dir'] = out_dir
    print('configs loaded')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    exp_name = os.path.basename(os.path.dirname(args.config))
    model = model_config.get_model(cfg, device=device)
    if args.parallel:
        model.encoder = torch.nn.DataParallel(model.encoder)
        model.decoder = torch.nn.DataParallel(model.decoder)

    model.eval()

    val_dataset = config.get_dataset('val', cfg)

    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    trainer = model_config.get_trainer(model, optimizer, cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    if args.it is not None:
        load_dict = checkpoint_io.load(f'model_{args.it}.pt')
    else:
        load_dict = checkpoint_io.load('model.pt')

    if args.sceneid_start is not None:
        for i in range(args.sceneid_start, args.sceneid_stop):
            process_scene(i)
    else:
        process_scene(args.sceneid)


