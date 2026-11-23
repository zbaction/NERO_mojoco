#!/usr/bin/env python3
"""诊断抓取的几何：沿 hover->grasp 关节线性插值，看指尖什么时候进方块、在哪一侧撞上。"""
from __future__ import annotations

import pathlib

import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
m = mujoco.MjModel.from_xml_path(str(HERE / "nero_table_scene.xml"))
d = mujoco.MjData(m)
ARM = ["joint%d" % i for i in range(1, 8)]
QAD = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM]
PADL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
PADR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")
CUBE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_green")
GL = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_l")]
GR = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_r")]

q_hover = np.array([m.key("hover").qpos[a] for a in QAD])
q_grasp = np.array([m.key("grasp").qpos[a] for a in QAD])
print("joint space delta (deg): " + " ".join(
    "%+.1f" % x for x in np.degrees(q_grasp - q_hover)))


def state(q, grip=None):
    d.qpos[QAD] = q
    if grip is not None:
        d.qpos[GL] = d.qpos[GR] = grip
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    pl = d.geom_xpos[PADL].copy()
    pr = d.geom_xpos[PADR].copy()
    # pad 内表面 y（朝另一指方向）
    l_in = pl[1] - 0.006   # finger_l 在 +y 侧，内表面往 -y
    r_in = pr[1] + 0.006
    return pl, pr, 0.5 * (pl + pr), l_in, r_in


pl0, pr0, mid0, li0, ri0 = state(q_hover, 0.050)
print("\nhover: padL %s padR %s" % (np.round(pl0, 4), np.round(pr0, 4)))
print("       内间隙 %.1f mm" % ((li0 - ri0) * 1000))
pl1, pr1, mid1, li1, ri1 = state(q_grasp, 0.050)
print("grasp: padL %s padR %s" % (np.round(pl1, 4), np.round(pr1, 4)))
print("       内间隙 %.1f mm" % ((li1 - ri1) * 1000))
print("       pad 跨度 z: L %.4f~%.4f  R %.4f~%.4f"
      % (pl1[2] - 0.015, pl1[2] + 0.015, pr1[2] - 0.015, pr1[2] + 0.015))

print("\n沿关节线性插值（无动力学，只做运动学）：")
print("  t     mid_x  mid_y  mid_z   gap_mm  padL_z  padR_z")
for i in range(21):
    t = i / 20.0
    q = q_hover + t * (q_grasp - q_hover)
    pl, pr, mid, li, ri = state(q, 0.050)
    flag = ""
    if abs(mid[0] - 0.180) > 0.02 or abs(mid[1]) > 0.02:
        flag = "  <- 侧向跑偏"
    print("  %.2f  %+.4f %+.4f %+.4f  %5.1f  %+.4f %+.4f%s"
          % (t, mid[0], mid[1], mid[2], (li - ri) * 1000, pl[2], pr[2], flag))
