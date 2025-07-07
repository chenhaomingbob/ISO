import pickle
import numpy as np
import iso.data.utils.fusion as fusion
# from projects.mmdet3d_plugin.datasets.pipelines.fusion import TSDFVolume, rigid_transform
import cv2

colors = np.array(
    [
        [22, 191, 206, 255],  # 0
        [214, 38, 40, 255],
        [43, 160, 43, 255],
        [158, 216, 229, 255],
        [114, 158, 206, 255],
        [204, 204, 91, 255],
        [255, 186, 119, 255],
        [147, 102, 188, 255],
        [30, 119, 181, 255],
        [188, 188, 33, 255],
        [255, 127, 12, 255],
        [196, 175, 214, 255],
        [153, 153, 153, 255],
    ]
)


empty_target_list = [
    "gathered_data/scene0067_01/00051.pkl",
    "gathered_data/scene0083_00/00095.pkl",
    "gathered_data/scene0128_00/00030.pkl",
    "gathered_data/scene0301_02/00098.pkl",
    "gathered_data/scene0301_02/00099.pkl",
    "gathered_data/scene0375_00/00046.pkl",
    "gathered_data/scene0375_00/00047.pkl",
    "gathered_data/scene0378_01/00027.pkl",
    "gathered_data/scene0448_01/00059.pkl",
    "gathered_data/scene0448_01/00060.pkl",
    "gathered_data/scene0448_01/00064.pkl",
    "gathered_data/scene0448_01/00066.pkl",
    "gathered_data/scene0505_01/00082.pkl",
    "gathered_data/scene0538_00/00043.pkl",
    "gathered_data/scene0538_00/00044.pkl",
    "gathered_data/scene0625_01/00001.pkl",
    "gathered_data/scene0642_01/00090.pkl",
    "gathered_data/scene0674_00/00052.pkl",
    "gathered_data/scene0684_00/00043.pkl",
    "gathered_data/scene0702_01/00027.pkl",
    "gathered_data/scene0702_01/00028.pkl",
    "gathered_data/scene0702_01/00029.pkl",
    "gathered_data/scene0702_01/00030.pkl",
    "gathered_data/scene0702_01/00031.pkl",
    # val set
    "gathered_data/scene0067_01/00052.pkl",
    "gathered_data/scene0121_01/00004.pkl",
    "gathered_data/scene0121_01/00005.pkl",
    "gathered_data/scene0128_00/00029.pkl",
    "gathered_data/scene0286_01/00087.pkl",
    "gathered_data/scene0301_02/00097.pkl",
    "gathered_data/scene0375_00/00048.pkl",
    "gathered_data/scene0448_01/00058.pkl",
    "gathered_data/scene0448_01/00061.pkl",
    "gathered_data/scene0448_01/00062.pkl",
    "gathered_data/scene0448_01/00063.pkl",
    "gathered_data/scene0448_01/00065.pkl",
    "gathered_data/scene0538_00/00083.pkl",
    "gathered_data/scene0674_00/00053.pkl"
]

def main():
    pc_range = [0, 0, 0, 4.8, 4.8, 2.4]
    # pc_range = [0, -2.4, 0, 4.8, 2.4, 2.4]
    # cam_k = np.array([[518.8579, 0, 320], [0, 518.8579, 240], [0, 0, 1]])
    # img_W = 640
    # img_H = 480
    scene_size = (4.8, 4.8, 2.88)
    voxel_size = 0.08

    pkl_file = "/data/chm/02_Dependices/mmdetection3d-1.3.0/data/occscannet/gathered_data/scene0067_01/00051.pkl"
    # pkl_file = "/data/chm/02_Dependices/mmdetection3d-1.3.0/data/occscannet/gathered_data/scene0031_02/00000.pkl"
    # pkl_file = "/data/chm/02_Dependices/mmdetection3d-1.3.0/data/occscannet/gathered_data/scene0002_00/00002.pkl"
    # img = "/data/chm/02_Dependices/mmdetection3d-1.3.0/data/scannet/posed_images/scene0002_00/00002.jpg"
    with open(pkl_file, 'rb') as handle:
        data = pickle.load(handle)
    vox_origin = np.array(list(data['voxel_origin']))
    # vox_origin = data['voxel_origin'] + np.array([0, 0.0, 0])
    # vox_origin = data['voxel_origin'] + np.array([0, -2.4, 0])
    # vox_origin = np.array([0,0,0])
    cam_pose = data["cam_pose"]
    voxel_sem = data['target_1_4']  # [60,60,36]
    voxel_sem = np.swapaxes(voxel_sem, 0, 1) # 这个为啥是反的？
    voxel_sem = voxel_sem.reshape(-1).astype(np.int32)
    # voxel_sem = voxel_sem.transpose(0, 2, 1).reshape(-1).astype(np.int32)
    T_world_2_cam = np.linalg.inv(cam_pose)
    # T_world_2_cam = np.linalg.inv(cam_pose)
    # project 3d point to 2d
    cam_intrin = data['intrinsic']
    img = cv2.imread(data['img'])
    img_H, img_W = img.shape[0], img.shape[1]
    W_factor = 640 / img_W
    H_factor = 480 / img_H
    cam_intrin[0, 0] *= W_factor
    cam_intrin[1, 1] *= H_factor
    cam_intrin[0, 2] *= W_factor
    cam_intrin[1, 2] *= H_factor

    pix_x, pix_y, pix_z, mask = vox2pix(T_world_2_cam, cam_intrin, img_H, img_W, scene_size, vox_origin, voxel_size)

    pix_x = pix_x[mask]
    pix_y = pix_y[mask]
    pix_z = pix_z[mask]
    voxel_sem = voxel_sem[mask]

    # image = cv2.imread(img)
    image = cv2.resize(img, (640, 480))
    # image = img
    # image = img
    for index, x in enumerate(pix_x):
        if voxel_sem[index] != 255 and voxel_sem[index] != 0:
            color = (
                int(colors[voxel_sem[index]][0]),
                int(colors[voxel_sem[index]][1]),
                int(colors[voxel_sem[index]][2])
            )

            cv2.circle(
                image,
                (int(pix_x[index]), int(pix_y[index])),  # (x,y)
                radius=2,
                color=color,
                thickness=-1
            )
            # cv2.circle(image, (int(pix_x[index]), int(pix_y[index])), radius=1, color=(0, 255, 0), thickness=-1)

    cv2.imshow('Projected Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vox2pix(T_world_2_cam, cam_k, img_H, img_W, scene_size, vox_origin, voxel_size):
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = vox_origin
    vol_bnds[:, 1] = vox_origin + np.array(scene_size)
    # vol_dim = np.ceil((np.around(vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size, 3)).copy(order='C').astype(int)
    # vol_dim = np.ceil((np.around(vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size, 3)).copy(order='C').astype(int)
    vol_dim = np.ceil(np.around((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size, 3)).copy(order='C').astype(int)
    if vol_dim[0] != 60 or vol_dim[1] != 60 or vol_dim[2] != 36:
        print("Find it:", vol_dim, '\n', vol_bnds, vol_bnds.dtype)
        exit(-1)
    xv, yv, zv = np.meshgrid(
        range(vol_dim[0]),
        range(vol_dim[1]),
        range(vol_dim[2]),
        indexing='ij'
    )
    vox_coords = np.concatenate([
        xv.reshape(1, -1),
        yv.reshape(1, -1),
        zv.reshape(1, -1)
    ], axis=0).astype(int).T  # 60x60x36 = 129600
    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)  # 每个voxel中心点在世界坐标系下的位置
    cam_pts = fusion.rigid_transform(cam_pts, T_world_2_cam)  # 每个voxel中心点在相机坐标下的位置
    # Project camera coordinates to pixel positions
    projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)  # 每个voxel中心点在图像上的像素位置
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    mask_camera = np.logical_and(pix_x >= 0,
                                 np.logical_and(pix_x < img_W,
                                                np.logical_and(pix_y >= 0,
                                                               np.logical_and(pix_y < img_H,
                                                                              pix_z > 0))))

    return pix_x, pix_y, pix_z, mask_camera


if __name__ == '__main__':
    main()
