#!/usr/bin/env python3
"""对比：静态 keyframe vs 动力学终态，夹爪偏航与方块偏航是否对齐。"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import mujoco

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402

m, d, qad = pp.load()
PADL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
PADR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")


def yaw_of_cube(bid):
    R = d.xmat[bid].reshape(3, 3)
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def yaw_of_pads():
    a = d.geom_xpos[PADL]
    b = d.geom_xpos[PADR]
    v = b - a
    # 夹爪 y 轴与方块 y 轴比较：把 v 折到 (-90,90]
    ang = np.degrees(np.arctan2(v[1], v[0])) - 90.0
    return (ang + 90.0) % 180.0 - 90.0


for col in ("red", "green", "blue"):
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_%s" % col)
    kf = "%s_grasp" % col
    d.qpos[:] = m.key(kf).qpos
    mujoco.mj_forward(m, d)
    ys, ps = yaw_of_cube(cid), yaw_of_pads()
    print("%-6s 静态: 方块偏航 %+7.2f°  夹爪偏航 %+7.2f°  差 %+7.2f°"
          % (col, ys, ps, ((ps - ys + 90) % 180) - 90))

print()
for col in ("red", "green", "blue"):
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_%s" % col)
    pp.reset(m, d, qad)
    pp.move(m, d, qad, "%s_hover" % col, pp.GRIP_OPEN, 2.0)
    q0 = np.array(d.qpos[qad])
    q1 = np.array([m.key("%s_grasp" % col).qpos[a] for a in qad])
    N = int(2.5 / m.opt.timestep)
    for i in range(N):
        s = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / N))
        d.ctrl[:7] = q0 + (q1 - q0) * s
        d.ctrl[7] = d.ctrl[8] = pp.GRIP_OPEN
        mujoco.mj_step(m, d)
    ys, ps = yaw_of_cube(cid), yaw_of_pads()
    print("%-6s 动态: 方块偏航 %+7.2f°  夹爪偏航 %+7.2f°  差 %+7.2f°   "
          "指尖 z=%.3f"
          % (col, ys, ps, ((ps - ys + 90) % 180) - 90,
             0.5 * (d.geom_xpos[PADL][2] + d.geom_xpos[PADR][2])))
