#!/usr/bin/env python3
"""测量静态 keyframe 下：夹爪开合方向 vs 方块自身 y 轴的夹角。

如果夹爪真正对齐方块，两指连线应该平行于方块局部 y 轴（夹角 0 或 180）。
"""
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

print("夹爪对齐诊断（静态 keyframe）")
print("  方块局部 y 轴 = 方块唯一的面法向；两指连线应与其平行")
for col in ("red", "green", "blue"):
    kf = "%s_grasp" % col
    d.qpos[:] = m.key(kf).qpos
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_%s" % col)
    R = d.xmat[bid].reshape(3, 3)
    ylocal = R[:, 1]          # 方块局部 +y 在世界系
    sep = d.geom_xpos[PADR] - d.geom_xpos[PADL]
    sep[2] = 0.0
    sep /= np.linalg.norm(sep)
    yl = ylocal.copy()
    yl[2] = 0.0
    yl /= np.linalg.norm(yl)
    ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(sep, yl))), 0, 1)))
    yaw_c = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    print("  %-6s 方块偏航 %+6.1f°  两指连线 [%+.3f %+.3f]  方块y [%+.3f %+.3f]"
          "  夹角 %6.1f°" % (col, yaw_c, sep[0], sep[1], yl[0], yl[1], ang))
    print("         期望：夹角 0（对齐）  实测/2 = %.1f° ≈ 方块偏航"
          % (ang / 2.0))
