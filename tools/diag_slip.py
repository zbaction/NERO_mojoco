#!/usr/bin/env python3
"""诊断搬运过程中的滑移：逐阶段记录方块相对夹爪的位姿偏移、接触数、方块姿态。"""
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

print("%-6s %-32s %-32s %-14s %s"
      % ("阶段", "方块(世界)", "夹爪指尖中点", "相对偏移(mm)", "接触/倾角"))
for i, (key, grip, secs, note) in enumerate(pp.PHASES):
    pp.move(m, d, qad, key, grip, secs)
    p = d.xpos[green].copy()
    pm = pp.pad_mid(m, d)
    R = d.xmat[EE].reshape(3, 3)
    # 方块在工具系里的位置（用 EE 旋转的逆）
    rel = R.T @ (p - pm)
    # 方块倾角：世界 z 轴与方块体 z 轴夹角
    Rb = d.xmat[green].reshape(3, 3)
    tilt = math.degrees(math.acos(max(-1, min(1, float(Rb[2, 2])))))
    con = 0
    for k in range(d.ncon):
        c = d.contact[k]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if "cube_green" in g1 + g2 and "finger" in g1 + g2:
            con += 1
    print("%-6d [%.3f %.3f %.3f]   [%.3f %.3f %.3f]   "
          "[%6.1f %6.1f %6.1f]   %d 接触 / 倾角 %4.1f°"
          % (i + 1, *p, *pm, 1000 * rel[0], 1000 * rel[1], 1000 * rel[2], con, tilt))
