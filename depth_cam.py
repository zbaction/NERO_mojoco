"""深度相机工具库：把 MuJoCo 的深度/分割渲染转成点云、并做几何测量。

用法（在 ~/NERO 下）：
    from depth_cam import DepthCamera
    dc = DepthCamera(m, d)                     # 默认读模型里的 wrist_cam
    dc = DepthCamera(m, d, cam_name="wrist_cam", width=848, height=480, max_depth=1.0)
    rgb, depth = dc.capture()
    pts, col, mask = dc.pointcloud(depth, rgb)   # 世界系点云（已剔除背景）

注意：`capture()` 返回的是 (rgb, depth)。老版本某些脚本按 pointcloud(rgb, depth)
顺序传参，内部会自动识别并纠正。

关键事实（tools/probe_depth*.py 实测，MuJoCo 3.17）：
  * mujoco.Renderer(enable_depth_rendering=True) 返回的是**沿相机光轴的 z**
    （不是欧氏距离），单位米，float32。
  * 背景（天空盒）像素的深度 = stat.extent * map.zfar（本场景 2.1*30 = 63.0），
    所以背景剔除用"深度 > 上限"而不是"非有限值"——实测所有像素都是有限值。
  * 内参：fy = 0.5*H/tan(fovy/2)，fx = fy（MuJoCo 用正方形像素）。
  * 相机坐标：x 右、y 上、z 向后（光轴 -z），像素 (u,v) 以左上角为原点，
    v 向下。反投影：xc=(u-W/2)/fx*z, yc=-(v-H/2)/fy*z, zc=-z。
  * **`<camera>` 挂在哪个 body 里就跟着哪个 body 运动**：本项目的 wrist_cam
    挂在 gripper_base 侧面，是 eye-in-hand；`pose()` 每次实时读
    `d.cam_xpos/d.cam_xmat`，所以不需要任何额外处理就能拿到当前外参。
"""
from __future__ import annotations

import numpy as np
import mujoco


class DepthCamera:
    """封装一台 MJCF 里定义的相机，提供 RGB / 深度 / 点云。"""

    def __init__(self, m, d, cam_name="depth_cam", width=None, height=None,
                 max_depth=2.5):
        self.m = m
        self.d = d
        self.name = cam_name
        self.cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cid < 0:
            raise ValueError("模型里没有相机 %r" % cam_name)
        self.fovy = float(m.cam_fovy[self.cid])
        # 默认分辨率：从 XML 的 offwidth/offheight 推断一个 16:9 的合理值
        self.width = int(width or 848)
        self.height = int(height or 480)
        self.max_depth = float(max_depth)
        self.fy = 0.5 * self.height / np.tan(np.radians(self.fovy) / 2.0)
        self.fx = self.fy
        self._r = mujoco.Renderer(m, self.height, self.width)

    # ---------------------------------------------------------------- 采集
    def capture(self, scene_option=None):
        """返回 (rgb uint8 HxWx3, depth float32 HxW)。"""
        self._r.disable_depth_rendering()
        self._r.enable_depth_rendering()
        self._r.update_scene(self.d, camera=self.cid, scene_option=scene_option)
        depth = self._r.render().copy()
        self._r.disable_depth_rendering()
        self._r.update_scene(self.d, camera=self.cid, scene_option=scene_option)
        rgb = self._r.render().copy()
        return rgb, depth

    def depth_only(self):
        self._r.enable_depth_rendering()
        self._r.update_scene(self.d, camera=self.cid)
        dep = self._r.render().copy()
        self._r.disable_depth_rendering()
        return dep

    # ---------------------------------------------------------------- 几何
    def intrinsics(self):
        """返回 (fx, fy, cx, cy)，可直接喂给 OpenCV 的 pinhole 模型。"""
        return self.fx, self.fy, self.width / 2.0, self.height / 2.0

    def pose(self):
        """返回 (pos(3,), R(3,3))：相机在世界系的位置与姿态（列 = 相机 xyz 轴）。"""
        R = np.array(self.d.cam_xmat[self.cid]).reshape(3, 3)
        return np.array(self.d.cam_xpos[self.cid]), R

    def pointcloud(self, depth, rgb=None, max_depth=None):
        """深度图 -> 世界系点云。返回 (pts Nx3, cols Nx3 uint8 或 None, mask)。

        mask 是与 depth 同形状的布尔图，标记"有效几何"（非天空/非过远）。
        """
        lim = self.max_depth if max_depth is None else float(max_depth)
        uu, vv = np.meshgrid(np.arange(self.width) + 0.5,
                             np.arange(self.height) + 0.5)
        zc = depth
        xc = (uu - self.width / 2.0) / self.fx * zc
        yc = -(vv - self.height / 2.0) / self.fy * zc
        pc = np.stack([xc, yc, -zc], axis=-1)
        p0, R = self.pose()
        pw = pc.reshape(-1, 3) @ R.T + p0
        mask = (depth > 0.05) & (depth < lim)
        pts = pw[mask.reshape(-1)]
        cols = rgb.reshape(-1, 3)[mask.reshape(-1)] if rgb is not None else None
        return pts, cols, mask

    def unproject(self, u, v, depth):
        """单个像素（可传数组）-> 世界坐标。"""
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        zc = np.asarray(depth, float)
        xc = (u - self.width / 2.0) / self.fx * zc
        yc = -(v - self.height / 2.0) / self.fy * zc
        pc = np.stack([xc, yc, -zc], axis=-1)
        p0, R = self.pose()
        return pc @ R.T + p0

    def project(self, pts):
        """世界坐标 (N,3) -> (u, v, depth_z)。在相机后方的点 depth_z <= 0。"""
        p0, R = self.pose()
        q = np.asarray(pts, float).reshape(-1, 3) - p0
        c = q @ R                       # 转到相机系
        zc = -c[:, 2]
        u = self.width / 2.0 + self.fx * c[:, 0] / zc
        v = self.height / 2.0 - self.fy * c[:, 1] / zc
        return u, v, zc

    def close(self):
        self._r.close()
