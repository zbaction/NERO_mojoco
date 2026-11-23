#!/usr/bin/env python3
"""检查 red/blue 的 keyframe 是否真的能到目标点（FK 校验 + 动力学跟踪）。"""
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

print("=== 1) 静态 FK：把 qpos 直接设成 keyframe，看指尖中点 ===")
for col in ("red", "green", "blue"):
    for ph in ("hover", "grasp", "lift", "transit", "place_hover", "place"):
        k = "%s_%s" % (col, ph)
        try:
            qp = m.key(k).qpos
        except Exception as e:
            print("%-16s 缺失 %s" % (k, e))
            continue
        d.qpos[:] = qp
        mujoco.mj_forward(m, d)
        pm = 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])
        print("  %-16s 指尖中点 [%.3f %.3f %.3f]" % (k, *pm))

print()
print("=== 2) 动力学跟踪：从 ready 平滑走到 <col>_grasp，看实际到达点 ===")
for col in ("red", "green", "blue"):
    pp.reset(m, d, qad)
    pp.move(m, d, qad, "%s_hover" % col, pp.GRIP_OPEN, 2.0)
    ph = pp.pad_mid(m, d).copy()
    pp.move(m, d, qad, "%s_grasp" % col, pp.GRIP_OPEN, 2.5)
    pg = pp.pad_mid(m, d).copy()
    # 目标
    d.qpos[:] = m.key("%s_grasp" % col).qpos
    mujoco.mj_forward(m, d)
    tgt = 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])
    print("  %-6s hover 实到 [%.3f %.3f %.3f]  grasp 实到 [%.3f %.3f %.3f]  "
          "目标 [%.3f %.3f %.3f]  误差 %.1f mm"
          % (col, *ph, *pg, *tgt, 1000 * np.linalg.norm(pg - tgt)))
