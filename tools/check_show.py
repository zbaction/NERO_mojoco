#!/usr/bin/env python3
"""统计展示图中机械臂的实际占比 + 亮度层次，并裁切放大看细节。"""
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


def drive(key, grip, secs):
    q = np.array([m.key(key).qpos[x] for x in qad])
    d.ctrl[:7] = q
    d.ctrl[7] = d.ctrl[8] = grip
    for _ in range(int(secs / m.opt.timestep)):
        mujoco.mj_step(m, d)


d.qpos[:] = m.key("ready").qpos
d.ctrl[:7] = [m.key("ready").qpos[x] for x in qad]
d.ctrl[7] = d.ctrl[8] = 0.05
mujoco.mj_forward(m, d)
drive("hover", 0.05, 2.0)
drive("grasp", 0.05, 2.5)
drive("grasp", 0.0, 2.0)
drive("hover", 0.0, 2.5)

m.vis.global_.fovy = 40
r = mujoco.Renderer(m, 720, 1280)
opt = mujoco.MjvOption()
opt.geomgroup[2] = 0

for name, az, el in (("showcase_grasp", 38, -14), ("showcase_front", 0, -6),
                     ("showcase_iso", 135, -20)):
    p = OUT / (name + ".png")
    if not p.exists():
        print("%s 缺失" % name)
        continue
    a = np.asarray(Image.open(p).convert("RGB")).astype(float)
    lum = a @ [0.2126, 0.7152, 0.0722]
    print("%-18s mean=%5.1f p5=%5.1f p50=%5.1f p95=%5.1f  过曝(>245)=%.2f%%  死黑(<15)=%.2f%%"
          % (name, lum.mean(), np.percentile(lum, 5), np.percentile(lum, 50),
             np.percentile(lum, 95), 100 * (lum > 245).mean(),
             100 * (lum < 15).mean()))

# 机械臂像素占比（用差分）
d.qpos[:] = m.key("grasp").qpos
mujoco.mj_forward(m, d)
cam = mujoco.MjvCamera()
cam.lookat[:] = (0.02, 0.0, 0.92)
cam.distance = 1.46
cam.azimuth = 38
cam.elevation = -14
r.update_scene(d, cam)
f1 = r.render()
r.update_scene(d, cam, opt)
f2 = r.render()
diff = np.abs(f1.astype(int) - f2.astype(int)).sum(axis=2) > 12
print("\n机械臂像素占比 %.2f%%" % (100 * diff.mean()))

# 裁切机械臂区域放大
ys, xs = np.nonzero(diff)
box = (max(0, xs.min() - 30), max(0, ys.min() - 30),
       min(1280, xs.max() + 30), min(720, ys.max() + 30))
im = Image.fromarray(r.render() if False else np.asarray(Image.open(
    OUT / "showcase_grasp.png").convert("RGB"))).crop(box)
im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
im.save(OUT / "z_show_crop.png")
print("裁切 %s -> %s" % (box, im.size))
r.close()
