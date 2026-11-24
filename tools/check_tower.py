"""探针：叠叠塔物理稳定性 —— 跑 5 s 物理看三层方块漂移与接触。"""
import os
import sys

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

m = mujoco.MjModel.from_xml_path(str(os.path.join(ROOT, "nero_table_scene.xml")))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

names = ["tower_red_geom", "tower_green_geom", "tower_blue_geom"]
gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
p0 = np.array([d.geom_xpos[g].copy() for g in gids])
print("初始顶面 z: %.4f" % (p0[-1][2] + 0.04))

for _ in range(int(5.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)

p1 = np.array([d.geom_xpos[g].copy() for g in gids])
dz = p1 - p0
print("5 s 后每层漂移 (mm): %s" % ", ".join(
    "%.2f" % (np.linalg.norm(v) * 1e3) for v in dz))
print("末层顶面 z: %.4f（理论 0.98）" % (p1[-1][2] + 0.04))
ncon = sum(1 for i in range(d.ncon) if d.contact[i].dist < 0)
print("接触对: %d" % ncon)
ok = all(np.linalg.norm(v) < 0.005 for v in dz)
print("STABLE" if ok else "UNSTABLE")
sys.stdout.flush()
os._exit(0 if ok else 1)
