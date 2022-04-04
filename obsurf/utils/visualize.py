import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from mpl_toolkits.mplot3d import Axes3D
import torch
from torchvision.utils import save_image

from colorsys import hsv_to_rgb



def background_image(shape, gridsize=2, lg=0.85, dg=0.5):
    bg = np.zeros(shape)
    c1 = np.array((lg, lg, lg))
    c2 = np.array((dg, dg, dg))

    for i, x in enumerate(range(0, shape[0], gridsize)):
        for j, y in enumerate(range(0, shape[1], gridsize)):
            c = c1 if (i + j) % 2 == 0 else c2
            bg[x:x+gridsize, y:y+gridsize] = c

    return bg


def make_rgba(images, alphas=None):
    if images.shape[-1] == 1:
        images = images.repeat(*([1] * (len(images.shape) - 1) + [3]))
    if alphas is not None:
        alphas = torch.clamp(alphas, 0., 1.)
        images = torch.cat((images, alphas.unsqueeze(-1)), -1)
    return images


def draw_bounding_box2d(img, corners):
    pil_img = Image.fromarray((img * 255.).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    num_points = corners.shape[-2]
    # Map world coords to pixels
    corners = (corners + 0.5) * np.expand_dims(np.array(img.shape[:2]), 0)

    # Flip x and y, since numpy/pytorch indexes rows first, whereas Pillow uses
    # a Cartesian coordinate system in which the first coordinate (x) is the horizontal.
    corners = [(corners[i][1], corners[i][0]) for i in range(num_points)]
    # Close the loop
    corners.append(corners[0])

    draw.line(corners, fill='green')

    img = np.array(pil_img).astype(np.float32) / 255.
    return img


def visualize_data(data, data_type, out_file):
    r''' Visualizes the data with regard to its type.

    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)


def visualize_2d_cluster(clustering, colors=None, outfile=None):
    if colors is None:
        num_clusters = clustering.max()
        colors = get_clustering_colors(num_clusters)
    img = colors[clustering]
    if outfile is not None:
        save_image(torch.tensor(np.transpose(img, (2, 0, 1))), outfile)
    return img


def get_clustering_colors(num_colors):
    colors = [(0., 0., 0.)]
    for i in range(num_colors):
        colors.append(hsv_to_rgb(i / num_colors, 0.45, 0.8))
    colors = np.array(colors)
    return colors


def setup_axis(axis):
    axis.tick_params(axis='both',       # changes apply to the x-axis
                     which='both',      # both major and minor ticks are affected
                     bottom=False,      # ticks along the bottom edge are off
                     top=False,         # ticks along the top edge are off
                     right=False,
                     left=False,
                     labelbottom=False,
                     labelleft=False)   # labels along the bottom edge are off


def draw_clustering_grid(columns, outfile, row_labels=None, name=None):
#def draw_clustering_grid(images, global_recons,
                         #pred_segmentations, local_recons,
                         #outfile, real_segmentation=None, labels=None):
    num_rows = columns[0][1].shape[0]
    num_cols = len(columns)
    num_segments = 1

    bg_image = None
    imshow_args = {'interpolation': 'none', 'cmap': 'gray'}

    for i in range(num_cols):
        column_type = columns[i][2]
        if column_type == 'clustering':
            num_segments = max(num_segments, columns[i][1].max())
        if column_type == 'image' and bg_image is None:
            bg_image = background_image(list(columns[i][1].shape[1:3]) + [3])

    colors = get_clustering_colors(num_segments)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2),
                             squeeze=False)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for c in range(num_cols):
        axes[0, c].set_title(columns[c][0])
        col_type = columns[c][2]
        for r in range(num_rows):
            setup_axis(axes[r, c])
            img = columns[c][1][r]
            if col_type == 'image':
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                axes[r, c].imshow(bg_image, **imshow_args)
                axes[r, c].imshow(img, **imshow_args)
                if len(columns[c]) > 3:
                    axes[r, c].set_xlabel(columns[c][3][r])
            elif col_type == 'clustering':
                axes[r, c].imshow(visualize_2d_cluster(img, colors), **imshow_args)

    if row_labels is not None:
        for r in range(num_rows):
            axes[r, 0].set_ylabel(row_labels[r])

    plt.savefig(f'{outfile}.png')
    plt.savefig(f'{outfile}.pdf')
    plt.close()

def visualize_voxels(voxels, out_file=None, show=False, colors=None):
    r''' Visualizes voxel data.

    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)

    color_names = np.array(['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'white', 'gray'])
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k', facecolors=color_names[colors])
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualise_projection(
        self, points, world_mat, camera_mat, img, output_file='out.png'):
    r''' Visualizes the transformation and projection to image plane.

        The first points of the batch are transformed and projected to the
        respective image. After performing the relevant transformations, the
        visualization is saved in the provided output_file path.

    Arguments:
        points (tensor): batch of point cloud points
        world_mat (tensor): batch of matrices to rotate pc to camera-based
                coordinates
        camera_mat (tensor): batch of camera matrices to project to 2D image
                plane
        img (tensor): tensor of batch GT image files
        output_file (string): where the output should be saved
    '''
    points_transformed = common.transform_points(points, world_mat)
    points_img = common.project_to_camera(points_transformed, camera_mat)
    pimg2 = points_img[0].detach().cpu().numpy()
    image = img[0].cpu().numpy()
    plt.imshow(image.transpose(1, 2, 0))
    plt.plot(
        (pimg2[:, 0] + 1)*image.shape[1]/2,
        (pimg2[:, 1] + 1) * image.shape[2]/2, 'x')
    plt.savefig(output_file)
