import os
import shutil
import sys
from pathlib import Path

import numpy as np
import sapien
import torch
from PIL import Image
import json
import natsort

import colorsys
from utils.spatial import rotation_matrix_to_axis, eliminate_x_component, eliminate_y_component, eliminate_z_component
import pdb

def spherical_to_cartesian(theta_phi):
    """ 将球面坐标 (θ,φ) 转换为三维笛卡尔坐标 """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=1)

def initialize_random_spherical_angles(N):
    """ 在球面上均匀初始化角度参数 """
    # θ ∈ [0, π] 使用反余弦保证均匀分布
    theta = torch.acos(2 * torch.rand(N, dtype=torch.float64) - 1)
    # φ ∈ [0, 2π) 均匀分布
    phi = torch.rand(N, dtype=torch.float64) * 2 * np.pi
    return torch.stack([theta, phi], dim=1).requires_grad_(True)

def initialize_uniform_equatorial_angles(N):
    """
    在赤道平面（XY平面）上生成均匀分布的固定角度参数
    - θ 固定为 90°（π/2）
    - φ 在 [0, 2π) 上等间距分布（非随机）
    """
    # θ 固定为 π/2（90度）
    theta = torch.full((N,), np.pi / 2, dtype=torch.float64)
    
    # φ 等间距分布：生成 [0, 2π) 的 N 个等分点
    phi = torch.linspace(0, 2 * np.pi, N, dtype=torch.float64)
    
    return torch.stack([theta, phi], dim=1).requires_grad_(True)

def thomson_random_spherical_solver(N, max_iter=200):
    # 初始化可优化角度参数
    angles = initialize_random_spherical_angles(N)
    optimizer = torch.optim.Adam([angles], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.96)

    best_energy = float('inf')
    best_angles = None

    for step in range(max_iter):
        optimizer.zero_grad()

        # 转换为三维坐标
        points = spherical_to_cartesian(angles)

        # 计算电势能
        pairwise_dist = torch.cdist(points, points)
        triu = torch.triu_indices(N, N, offset=1)
        distances = pairwise_dist[triu[0], triu[1]]
        energy = torch.sum(1 / (distances + 1e-12))  # 添加极小值防止除零

        # 反向传播
        energy.backward()
        optimizer.step()
        scheduler.step()

        # 记录最佳解
        if energy < best_energy:
            best_energy = energy.item()
            best_angles = angles.detach().clone()

    # 转换为最终坐标
    best_points = spherical_to_cartesian(best_angles)
    return best_points.numpy(), best_energy

def thomson_uniform_equatorial_solver(N, max_iter=200):
    # 初始化可优化角度参数
    angles = initialize_uniform_equatorial_angles(N) 
    
    best_energy = float('inf')
    best_energy = None
    
    # 转换为最终坐标
    best_points = spherical_to_cartesian(angles.detach().clone())
    return best_points.numpy(), best_energy

VIEWS_CNT: int = 9
EXTRA_POSE_CNT: int = 5


def check_model(model_id, loader):
    model_root = os.path.join("partnet-mobility-dataset", str(model_id))
    try:
        robot = loader.load(os.path.join(model_root, "mobility.urdf"))
        robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    except (RuntimeError, ValueError) as e:
        print(f"Removing model {model_id} due to error:\n{e}")
        shutil.move(model_root, ".")
        return


def get_cam_pose(cam_pos) -> sapien.Pose:
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos

    return sapien.Pose(mat44)


def find_alpha_bounding_box(rgba_img):
    if rgba_img.shape[2] != 4:
        raise ValueError("The image must have 4 channels (RGBA).")
    
    alpha_channel = rgba_img[:, :, 3]
    non_zero_alpha = np.nonzero(alpha_channel)
    
    if non_zero_alpha[0].size == 0:
        return None
    
    min_row, max_row = np.min(non_zero_alpha[0]), np.max(non_zero_alpha[0])
    min_col, max_col = np.min(non_zero_alpha[1]), np.max(non_zero_alpha[1])
    
    return (min_row.item(), min_col.item(), max_row.item(), max_col.item())


def generate_distinct_colors(n):
    """生成N个醒目的不同颜色（HSV转换到RGB，固定饱和度和亮度）"""
    colors = []
    for i in range(n):
        hue = i / n  # 色调等间距
        saturation = 1.0  # 最大饱和度
        value = 1.0  # 最大亮度
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb = tuple(int(c * 255) for c in rgb)
        colors.append((*rgb, 255))  # 转为RGBA，Alpha=255
    return colors

def draw_bboxes_on_image(rgba_img, bboxes):
    """
    在图像上绘制多个不同颜色的边界框
    :param rgba_img: 原始图像（形状为[H, W, 4]的numpy数组）
    :param bboxes: 边界框列表，每个元素为(min_row, min_col, max_row, max_col)
    :return: 绘制后的图像副本（numpy数组）
    """
    # 创建图像副本避免修改原图
    height, width, _ = rgba_img.shape
    
    # 生成不同颜色
    num_boxes = len(bboxes)
    colors = generate_distinct_colors(num_boxes)
    
    for i, (key, bbox) in enumerate(bboxes.items()):
        min_row, min_col, max_row, max_col = bbox
        color = colors[i]
        
        # 确保坐标在图像范围内
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(height - 1, max_row)
        max_col = min(width - 1, max_col)
        
        # 绘制上边框和下边框
        if min_row <= max_row and min_col <= max_col:
            rgba_img[min_row, min_col:max_col+1, :] = color
            rgba_img[max_row, min_col:max_col+1, :] = color
        
        # 绘制左边框和右边框
        if min_col <= max_col and min_row <= max_row:
            rgba_img[min_row:max_row+1, min_col, :] = color
            rgba_img[min_row:max_row+1, max_col, :] = color
    
    return


def generate_samples(min_lim, max_lim, s):
    """
    Generate s arrays (samples) where for each dimension,
    candidate values are evenly spaced between min_lim and max_lim (excluding the limits),
    and each sample randomly picks one of these candidates without replacement.

    Parameters:
        min_lim (array-like): 1D array of minimum limits per dimension.
        max_lim (array-like): 1D array of maximum limits per dimension.
        s (int): Number of samples (and number of candidate values per dimension).

    Returns:
        np.ndarray: Array of shape (s, n_dims) where each row is a sample.
    """
    min_lim = np.asarray(min_lim)
    max_lim = np.asarray(max_lim)

    n_dims = min_lim.shape[0]

    candidate_samples = [
        np.linspace(min_lim[i], max_lim[i], num=s)
        for i in range(n_dims)
    ]

    samples = np.empty((s, n_dims))

    for i in range(n_dims):
        permuted = np.random.permutation(candidate_samples[i])
        samples[:, i] = permuted

    return samples


class ModelRenderer:
    scene = None
    camera = None

    def __init__(self):
        self.scene = sapien.Scene()  # Create an instance of simulation world (aka scene)
        self.scene.set_timestep(1 / 100.0)  # Set the simulation frequency

        # Add some lights so that you can observe the scene
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        ##### Setup camera
        near, far = 0.1, 100
        width, height = 560, 560

        self.camera = self.scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )

    def render_pos_inner(self):
        self.scene.update_render()  # Update the world to the renderer

        suc = False
        while not suc:
            try:
                self.camera.take_picture()
                suc = True
            except RuntimeError:
                pass

        rgba = self.camera.get_picture("Color")  # [H, W, 4]
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        return rgba_img
    def render_pose(self, robot, pos, dest):
        robot.set_qpos(pos)
        
        components_with_names = []
        for e in self.scene.entities:
            c = e.find_component_by_type(sapien.pysapien.render.RenderBodyComponent)
            if c is not None:
                name = e.get_name()
                components_with_names.append((c, name))
                        
        for c,_ in components_with_names:
            c.disable()
            
        bboxes = {}
        for c, name in components_with_names:
            c.enable()
            img = self.render_pos_inner()
            if (bbox := find_alpha_bounding_box(img)):
                bboxes[name] = bbox
            c.disable()
            
        for c,_ in components_with_names:
            c.enable()
            
        with dest.with_name(dest.stem + "_bbox.json").open('w') as f:
            json.dump(bboxes, f)
            
        rgba_img =  self.render_pos_inner()
        rgba_pil = Image.fromarray(rgba_img)
        rgba_pil.save(dest.with_name(dest.stem + "_raw.png"))
        
        draw_bboxes_on_image(rgba_img, bboxes)
        rgba_pil = Image.fromarray(rgba_img)
        rgba_pil.save(dest.with_name(dest.stem + "_bbox.png"))

    def load_model(self, model_id: str) -> sapien.physx.PhysxArticulation:
        while len(self.scene.get_all_articulations()) != 0:
            self.scene.remove_articulation(self.scene.get_all_articulations()[0])

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True

        robot = loader.load(os.path.join("partnet-mobility-dataset", str(model_id), "mobility.urdf"))
        robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        for link in robot.links:
            link.disable_gravity = True
        return robot

    def process(self, model_id: str):
        print(f"Processing model {model_id}")
        dest = Path("rendered") / str(model_id)
        dest.mkdir(exist_ok=True)

        robot = self.load_model(model_id)

        limits = robot.get_qlimits()
        limits = np.where(limits == np.inf, 2, limits)
        limits = np.where(limits == -np.inf, -2, limits)

        min_lim, max_lim = limits.T
        model_poses = [min_lim, max_lim]
        model_poses.extend(generate_samples(min_lim, max_lim, EXTRA_POSE_CNT))

        # views = (lambda res: [res[0][i] * 4 for i in range(VIEWS_CNT)])(thomson_random_spherical_solver(VIEWS_CNT))
        views = (lambda res: [res[0][i] * 4 for i in range(VIEWS_CNT)])(thomson_uniform_equatorial_solver(VIEWS_CNT))
        # R = rotation_matrix_to_axis(views[0], "x+")
        # views = [R @ v for v in views]
        
        # for id, v in enumerate(views):
        #     if id == 0:
        #         R = rotation_matrix_to_axis(v, "x+")
        #     else:
        #         R = eliminate_z_component(v)
        #     views[id] = R @ v
        
        for pose_idx in range(len(model_poses)):
            for view_idx in range(len(views)):
                self.camera.entity.set_pose(get_cam_pose(views[view_idx]))
                dest_file = dest / f"Pose_{pose_idx}_View_{view_idx}"
                self.render_pose(robot, model_poses[pose_idx], dest_file)
                
        print(f"Finished processing model {model_id}")
        
        camera_json = dest / "camera.json"
        with camera_json.open('w') as f:
            json.dump([v.tolist() for v in views], f)

    def handle_stdin(self):
        for line in iter(sys.stdin.readline, ""):
            self.process(line.strip())


if __name__ == "__main__":
    # all_model_ids = os.listdir("partnet-mobility-dataset")
    all_model_ids = ['5850']
    r = ModelRenderer()
    r.process(all_model_ids[0])

    # with Pool(192) as p:
    #    list(p.imap_unordered(process_model, all_model_ids))

    # for m in all_model_ids:
    #    print(f"Processing model {m}")
    #    process_model(m)