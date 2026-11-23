#!/usr/bin/env python3
"""验证：把夹爪开度从 0.050 提到不同值，蓝色能否降到目标高度。"""
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

TARGET = {"red": np.array([0.160, -0.260, 0.780]),
          "green": np.array([0.180, 0.000, 0.780]),
          "blue": np.array([0.150, 0.260, 0.780])}


def tip():
    return 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])


for col in ("red", "green", "blue"):
    print("=== %s  目标 %s ===" % (col, np.round(TARGET[col], 3)))
    for op in (0.050, 0.052, 0.053, 0.055):
        pp.reset(m, d, qad)
        pp.move(m, d, qad, "%s_hover" % col, pp.GRIP_OPEN, 2.0)
        q0 = np.array(d.qpos[qad])
        q1 = np.array([m.key("%s_grasp" % col).qpos[a] for a in qad])
        N = int(2.5 / m.opt.timestep)
        for i in range(N):
            s = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / N))
            d.ctrl[:7] = q0 + (q1 - q0) * s
            d.ctrl[7] = d.ctrl[8] = op
            mujoco.mj_step(m, d)
        t = tip()
        perr = 1000.0 * np.linalg.norm(t - TARGET[col])
        jerr = np.degrees(np.abs(d.qpos[qad] - q1)).max()
        sat = np.abs(d.actuator_force[:7]) >= 23.9
        print("  开度 %.3f (%3.0f mm)  指尖 %s  err %6.1f mm  "
              "最大关节误差 %5.2f°  %s"
              % (op, 2000 * op, np.round(t, 4), perr, jerr,
                 "力矩饱和" if sat.any() else "ok"))
