import numpy as np
from mayavi import mlab
import argparse


def load_voxels(path):
    """Load voxel labels from file.

    Args:
        path (str): The path of the voxel labels file.

    Returns:
        ndarray: The voxel labels with shape (N, 4), 4 is for [x, y, z, label].
    """
    labels = np.load(path)
    if labels.shape[1] == 7:
        labels = labels[:, [0, 1, 2, 6]]

    return labels


def draw(voxel_label, voxel_size=0.05, intrinsic=None, cam_pose=None, d=0.5, vox_origin=None):
    """Visualize the gt or predicted voxel labels.

    Args:
        voxel_label (ndarray): The gt or predicted voxel label, with shape (N, 4), N is for number
            of voxels, 7 is for [x, y, z, label].
        voxel_size (double): The size of each voxel.
        intrinsic (ndarray): The camera intrinsics.
        cam_pose (ndarray): The camera pose.
        d (double): The depth of camera model visualization.
    """
    figure = mlab.figure(size=(1600 * 0.8, 900 * 0.8), bgcolor=(1, 1, 1))

    if intrinsic is not None and cam_pose is not None and vox_origin is not None:
        assert d > 0, 'camera model d should > 0'
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # half of the image plane size
        y = d * 2 * cy / (2 * fy)
        x = d * 2 * cx / (2 * fx)

        # camera points (cam frame)
        tri_points = np.array(
            [
                [0, 0, 0],
                [x, y, d],
                [-x, y, d],
                [-x, -y, d],
                [x, -y, d],
            ]
        )
        tri_points = np.hstack([tri_points, np.ones((5, 1))])

        # camera points (world frame)
        tri_points = (cam_pose @ tri_points.T).T
        x = tri_points[:, 0] - vox_origin[0]
        y = tri_points[:, 1] - vox_origin[1]
        z = tri_points[:, 2] - vox_origin[2]
        triangles = [
            (0, 1, 2),
            (0, 1, 4),
            (0, 3, 4),
            (0, 2, 3),
        ]

        # draw cam model
        mlab.triangular_mesh(
            x,
            y,
            z,
            triangles,
            representation="wireframe",
            color=(0, 0, 0),
            line_width=7.5,
        )

    # draw occupied voxels
    plt_plot = mlab.points3d(
        voxel_label[:, 0],
        voxel_label[:, 1],
        voxel_label[:, 2],
        voxel_label[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.1 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=12,
    )

    # label colors
    colors = np.array(
        [
            [0, 0, 0, 255],  # 0 empty
            [255, 202, 251, 255],  # 1 ceiling
            [208, 192, 122, 255],  # 2 floor
            [199, 210, 255, 255],  # 3 wall
            [82, 42, 127, 255],  # 4 window
            [224, 250, 30, 255],  # 5 chair
            [255, 0, 65, 255],  # 6  bed
            [144, 177, 144, 255],  # 7 sofa
            [246, 110, 31, 255],  # 8 table
            [0, 216, 0, 255],  # 9 tv
            [135, 177, 214, 255],  # 10 furniture
            [1, 92, 121, 255],  # 11 objects
            [128, 128, 128, 255],  # 12 occupied with semantic
        ]
    )

    plt_plot.glyph.scale_mode = "scale_by_vector"

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    mlab.show()


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)

    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def parse_args():
    parser = argparse.ArgumentParser(description="CompleteScanNet dataset visualization.")
    parser.add_argument("--file", type=str, help="Voxel label file path.", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.file.endswith(".npy"):
        voxels = load_voxels(args.file)
        valid_mask = np.logical_and(voxels[:, -1] > 0, voxels[:, -1] < 255)
        voxels = voxels[valid_mask]
        draw(voxels, voxel_size=0.08, d=0.5)
    else:
        import pickle

        with open(args.file, "rb") as handle:
            b = pickle.load(handle)

        voxel_size = 0.08
        cam_pose = b["cam_pose"]
        # 交换了target的
        if 'target_1_4' in b:
            # 读的gt
            target = b['target_1_4']  # (60,60,36)
            target = np.swapaxes(target, 0, 1)
            intrinsic = b['intrinsic']
        else:
            # 读的预测
            cam_pose = b["cam_pose"]
            cam_pose = np.linalg.inv(cam_pose)
            # target = b['target']
            # target = np.swapaxes(target, 0, 1)
            target = b['pred']
            intrinsic = b['cam_intrinsic']

        voxel_origin = np.array(b["voxel_origin"])
        # temp = voxel_origin[1]
        # voxel_origin[1] = voxel_origin[0]
        # voxel_origin[0] = temp
        grid_coords = get_grid_coords(
            [target.shape[0], target.shape[1], target.shape[2]], voxel_size
        )
        voxels = np.vstack((grid_coords.T, target.reshape(-1))).T
        valid_mask = np.logical_and(voxels[:, -1] > 0, voxels[:, -1] < 255)
        voxels = voxels[valid_mask]
        # intrinsic = b['intrinsic']
        voxels[:, :3] = voxels[:, :3]
        draw(voxels, voxel_size=voxel_size, intrinsic=intrinsic, cam_pose=cam_pose, d=0.5, vox_origin=voxel_origin)
