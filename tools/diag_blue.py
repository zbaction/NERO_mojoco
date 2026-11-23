#!/usr/bin/env python3
"""记录 blue hover->grasp 的逐帧关节误差，找出被什么卡住。"""
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
JN = ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]


def tip():
    return 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])


col = "blue"
pp.reset(m, d, qad)
pp.move(m, d, qad, "%s_hover" % col, pp.GRIP_OPEN, 2.0)
q0 = np.array(d.qpos[qad])
q1 = np.array([m.key("%s_grasp" % col).qpos[a] for a in qad])
print("hover qpos: " + " ".join("%+7.1f" % np.degrees(v) for v in q0))
print("grasp qpos: " + " ".join("%+7.1f" % np.degrees(v) for v in q1))
print("需要扫过  : " + " ".join("%+7.1f" % v for v in np.degrees(q1 - q0)))
print()
print("关节限位余量（grasp 目标）:")
for i, jn in enumerate(JN):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % (i + 1))
    lo, hi = m.jnt_range[jid]
    print("  %s %.1f°  [%.1f, %.1f]  余量 %.1f°"
          % (jn, np.degrees(q1[i]), np.degrees(lo), np.degrees(hi),
             np.degrees(min(q1[i] - lo, hi - q1[i]))))

print()
print("逐步余弦插值，记录轨迹：")
N = int(2.5 / m.opt.timestep)
dt = m.opt.timestep
for i in range(N):
    s = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / N))
    d.ctrl[:7] = q0 + (q1 - q0) * s
    d.ctrl[7] = d.ctrl[8] = pp.GRIP_OPEN
    mujoco.mj_step(m, d)
    if (i + 1) % max(1, N // 10) == 0:
        e = np.degrees(d.qpos[qad] - d.ctrl[:7])
        print("  step %4d  tip=[%.3f %.3f %.3f]  err=%s"
              % (i + 1, *tip(), " ".join("%+5.1f" % v for v in e)))
print()
e = np.degrees(d.qpos[qad] - q1)
print("终态误差: " + " ".join("%+6.2f" % v for v in e))
print("终态指尖: %s   目标 [%.3f %.3f %.3f]"
      % (np.round(tip(), 3), 0.150, 0.260, 0.780))
print("cvel 末端: %s" % np.round(d.cvel[mujoco.mj_name2id(
    m, mujoco.mjtObj.mjOBJ_BODY, "end_effector")], 3))
