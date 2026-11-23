#!/usr/bin/env python3
"""自动找机械臂像素的包围盒，裁出贴合构图的特写，并统计细节（是否为纯剪影）。"""
import pathlib

import numpy as np
import mujoco
from PIL import Image

HERE = pathlib.Path.home() / "NERO"
OUT = HERE / "out"
m = mujoco.MjModel.from_xml_path(str(HERE / "nero_table_scene.xml"))
d = mujoco.MjData(m)
ARM = ["joint%d" % i for i in range(1, 8)]
qad = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM]

d.qpos[:] = m.key("grasp").qpos
d.ctrl[:7] = [m.key("grasp").qpos[x] for x in qad]
d.ctrl[7] = d.ctrl[8] = 0.0
mujoco.mj_forward(m, d)

m.vis.global_.fovy = 38
r = mujoco.Renderer(m, 720, 1280)
cam = mujoco.MjvCamera()
cam.lookat[:] = (0.02, 0.0, 0.94)
cam.distance = 1.05
cam.azimuth = 40
cam.elevation = -14

r.update_scene(d, cam)
full = r.render()
Image.fromarray(full).save(OUT / "z_probe_full.png")

# 关掉机械臂的可见性组，做差分找机械臂像素
opt = mujoco.MjvOption()
opt.geomgroup[2] = 0
r.update_scene(d, cam, opt)
noarm = r.render()

diff = np.abs(full.astype(int) - noarm.astype(int)).sum(axis=2) > 12
ys, xs = np.nonzero(diff)
print("机械臂像素 %d（%.2f%%） 包围盒 x[%d,%d] y[%d,%d]"
      % (diff.sum(), 100 * diff.mean(), xs.min(), xs.max(), ys.min(), ys.max()))

pad = 40
box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
       min(1280, xs.max() + pad), min(720, ys.max() + pad))
im = Image.fromarray(full).crop(box)
im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
im.save(OUT / "z_probe_crop.png")
print("裁切 %s -> %s" % (box, im.size))

# 细节检验：机械臂区域内的颜色多样性
reg = full[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
col = reg.reshape(-1, 3)
uniq = len(np.unique((col // 8).astype(np.uint8), axis=0))
lum = reg.astype(float) @ [0.2126, 0.7152, 0.0722]
print("机械臂区域：量化颜色数=%d  亮度 mean=%.1f std=%.1f min=%.0f max=%.0f"
      % (uniq, lum.mean(), lum.std(), lum.min(), lum.max()))
r.close()
