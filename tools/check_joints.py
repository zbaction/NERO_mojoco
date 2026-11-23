#!/usr/bin/env python3
"""确认关节在 qpos 里的排布，避免硬编码索引出错。"""
from __future__ import annotations

import pathlib
import sys

import mujoco

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402

m, d, qad = pp.load()
print("nq=%d  nv=%d  njnt=%d" % (m.nq, m.nv, m.njnt))
print()
print("%-3s %-22s %-10s %-8s %s" % ("id", "joint", "type", "qposadr", "dofadr"))
for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    t = m.jnt_type[j]
    tname = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(int(t), str(t))
    print("%-3d %-22s %-10s %-8d %d"
          % (j, name, tname, m.jnt_qposadr[j], m.jnt_dofadr[j]))
print()
print("机械臂关节 qpos 地址:", qad)
print("手指关节:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                    for j in range(m.njnt) if "grip" in
                    (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")])
