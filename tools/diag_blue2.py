#!/usr/bin/env python3
"""blue grasp 终态到底卡在什么上：打印所有接触 + 各 geom 世界坐标。"""
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
CUBE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_blue")

col = "blue"
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

print("=== 终态 geom 世界坐标 ===")
for g in (PADL, PADR):
    print("  %-16s pos=%s  size=%s" % (
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
        np.round(d.geom_xpos[g], 4), np.round(m.geom_size[g], 4)))
print("  cube_blue centre=%s  half=0.040  -> x[%.3f %.3f] y[%.3f %.3f] z[%.3f %.3f]"
      % (np.round(d.xpos[CUBE], 4),
         d.xpos[CUBE][0] - 0.04, d.xpos[CUBE][0] + 0.04,
         d.xpos[CUBE][1] - 0.04, d.xpos[CUBE][1] + 0.04,
         d.xpos[CUBE][2] - 0.04, d.xpos[CUBE][2] + 0.04))

print()
print("=== 全部接触（ncon=%d）===" % d.ncon)
for k in range(d.ncon):
    c = d.contact[k]
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?%d" % c.geom1
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?%d" % c.geom2
    if "cube" in g1 or "cube" in g2 or "finger" in g1 or "finger" in g2:
        print("  %-22s <-> %-22s dist=%.5f" % (g1, g2, c.dist))
print()
print("=== 关节执行器力 ===")
for i in range(7):
    print("  joint%d  force=%+8.2f  range=[%.0f %.0f]"
          % (i + 1, d.actuator_force[i],
             m.actuator_forcerange[i][0], m.actuator_forcerange[i][1]))
