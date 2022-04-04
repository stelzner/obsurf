import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import imageio
import numpy as np

import argparse, os, subprocess
from os.path import join

from obsurf.utils.visualize import setup_axis, background_image

def compile_video(path, small=False, frames=False):

    frame_output_dir = os.path.join(path, 'frames_small' if small else 'frames')
    if not os.path.exists(frame_output_dir):
        os.mkdir(frame_output_dir)

    input_image = imageio.imread(join(path, 'input.png'))
    bg_image = background_image((240, 320, 3), gridsize=12)

    frame_id = 0
    dpi = 100

    while frames:
        if small:
            fig, ax = plt.subplots(2, 2, figsize=(600/dpi, 480/dpi), dpi=dpi)
        else:
            fig, ax = plt.subplots(3, 4, figsize=(1280/dpi, 720/dpi), dpi=dpi)

        plt.subplots_adjust(wspace=0.05, hspace=0.08, left=0.01, right=0.99, top=0.995, bottom=0.035)
        for row in ax:
            for cell in row:
                setup_axis(cell)

        ax[0, 0].imshow(input_image)
        ax[0, 0].set_xlabel('Input Image')
        try:
            render = imageio.imread(join(path, 'renders', f'{frame_id}.png'))
        except FileNotFoundError:
            break
        print(f'{frame_id}...')
        ax[0, 1].imshow(bg_image)
        ax[0, 1].imshow(render[..., :3])
        ax[0, 1].set_xlabel('Rendered Scene')
        depths = imageio.imread(join(path, 'depths', f'{frame_id}.png'))
        if small:
            depths = depths.astype(np.float32) / 65536.
            ax[1, 0].imshow(depths, cmap='viridis')
            ax[1, 0].set_xlabel('Render Depths')
        else:
            depths = 1. - depths.astype(np.float32) / 65536.
            ax[0, 2].imshow(depths, cmap='viridis')
            ax[0, 2].set_xlabel('Render Depths')
        segmentations = imageio.imread(join(path, 'segmentations', f'{frame_id}.png'))
        if small:
            ax[1, 1].imshow(segmentations)
            ax[1, 1].set_xlabel('Segmentations')
        else:
            ax[0, 3].imshow(segmentations)
            ax[0, 3].set_xlabel('Segmentations')

        if small:
            fig.savefig(join(frame_output_dir, f'{frame_id}.png'))
            plt.close()

            frame_id += 1
            continue

        for slot_id in range(8):
            row = 1 + slot_id // 4
            col = slot_id % 4
            try:
                slot_render = imageio.imread(join(path, 'slot_renders', f'{slot_id}-{frame_id}.png'))
            except FileNotFoundError:
                ax[row, col].axis('off')
                continue
            # if (slot_render[..., 3] > 0.1).astype(np.float32).mean() < 0.4:
            ax[row, col].imshow(bg_image)
            ax[row, col].imshow(slot_render)
            ax[row, col].set_xlabel(f'Rendered Slot #{slot_id}')

        fig.savefig(join(frame_output_dir, f'{frame_id}.png'))
        plt.close()

        frame_id += 1

    frame_placeholder = join(frame_output_dir, '%d.png')
    video_out_file = join(path, 'video-small.mp4' if small else 'video.mp4')
    print('rendering video to ', video_out_file)
    subprocess.call(['ffmpeg', '-y', '-framerate', '60', '-i', frame_placeholder,
                     '-pix_fmt', 'yuv420p', '-crf', '17', '-threads', '1', video_out_file])


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Render a video of a scene.'
    )
    parser.add_argument('path', type=str, help='Path to image files.')
    parser.add_argument('--small', action='store_true', help='Create small 2x2 video.')
    parser.add_argument('--noframes', action='store_true', help="Assume frames already exist and don't rerender them.")
    args = parser.parse_args()

    compile_video(args.path, small=args.small, frames=not args.noframes)



