#!/usr/bin/env python3
"""定位滑移时刻：抓取闭合后，逐帧记录方块相对两指中点的位移，找出突变点。"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import mujoco

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402

m, d, qad = pp.load()
pp.reset(m, d, qad)
green = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_green")
EE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "end_effector")

pp.move(m, d, qad, "hover", pp.GRIP_OPEN, 2.0)
pp.move(m, d, qad, "grasp", pp.GRIP_OPEN, 2.5)
pp.move(m, d, qad, "grasp", pp.GRIP_CLOSE, 1.5)
base = (d.xpos[green] - pp.pad_mid(m, d)).copy()
print("闭合后基准相对偏移 %s mm" % np.round(1000 * base, 2))
print("方块中心 %s   pad 中点 %s" % (np.round(d.xpos[green], 4), np.round(pp.pad_mid(m, d), 4)))

for key, secs in [("lift", 2.0), ("transit", 2.5), ("place_hover", 2.5)]:
    q = np.array([m.key(key).qpos[a] for a in qad])
    d.ctrl[:7] = q
    d.ctrl[7] = d.ctrl[8] = pp.GRIP_CLOSE
    n = int(secs / m.opt.timestep)
    print("\n=== %s (%d 步) ===" % (key, n))
    prev = None
    worst = (0.0, -1)
    for i in range(n):
        mujoco.mj_step(m, d)
        rel = d.xpos[green] - pp.pad_mid(m, d)
        if i % 100 == 0 or i == n - 1:
            print("  step %4d  t=%.2fs  相对偏移 %s mm  |d|=%.1f  z方块=%.4f"
                  % (i, i * m.opt.timestep, np.round(1000 * rel, 1),
                     np.linalg.norm(rel) * 1000, d.xpos[green][2]))
        dr = np.linalg.norm(rel - base)
        if dr > worst[0]:
            worst = (dr, i)
    print("  >>> 本段最大相对位移 %.1f mm @ step %d" % (1000 * worst[0], worst[1]))
