#!/usr/bin/env python3
"""静态保持测试：把 qpos 设成 keyframe，ctrl 设成同一目标，看会不会被重力压走。
同时打印保持该姿态所需的关节力矩。"""
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


def tip():
    return 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])


KEYS = ["ready"] + ["%s_%s" % (c, p) for c in ("red", "green", "blue")
                    for p in ("hover", "grasp")]
for k in KEYS:
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.key(k).qpos
    mujoco.mj_forward(m, d)
    # 关掉重力补偿？不，保持现状，只测伺服能力
    q_t = np.array(d.qpos[qad])
    d.ctrl[:7] = q_t
    d.ctrl[7] = d.ctrl[8] = m.key(k).qpos[7]
    t0 = tip().copy()
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    t1 = tip().copy()
    # 保持时的执行器力
    mujoco.mj_forward(m, d)
    f = d.actuator_force[:7].copy()
    sat = np.abs(f) >= 23.9
    print("%-14s 落脚漂移 %5.1f mm  关节力 %s %s"
          % (k, 1000 * np.linalg.norm(t1 - t0),
             " ".join("%+6.1f" % v for v in f),
             " 饱和:" + ",".join("j%d" % (i + 1) for i in np.where(sat)[0])
             if sat.any() else ""))
