#!/usr/bin/env python3
"""确认关键帧 qpos 的长度语义 + 篮子候选位姿可达性。"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import mujoco

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))

m = mujoco.MjModel.from_xml_path(str(HERE / "nero_table_scene.xml"))
d = mujoco.MjData(m)
print("nq =", m.nq, " nv =", m.nv, " nkey =", m.nkey)
k = m.key("ready")
print("key qpos 长度 =", len(k.qpos))
print("前 9 个（臂+指）:", np.round(k.qpos[:9], 4))
print("其余（3 个方块 freejoint）:", np.round(k.qpos[9:], 4))
for n in ("cube_red", "cube_green", "cube_blue"):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    j = m.body_jntadr[b]
    a = m.jnt_qposadr[j]
    print("  %-11s qpos@%d = %s" % (n, a, np.round(k.qpos[a:a + 7], 4)))
