#!/usr/bin/env python3
"""逐步对比：把 qpos 设成 keyframe vs 真的伺服过去，指尖位置为何不同。"""
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
JOINTS = ["joint%d" % i for i in range(1, 8)]


def tip():
    return 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])


col = "blue"
kf = "%s_grasp" % col
q_cmd = np.array([m.key(kf).qpos[a] for a in qad])

pp.reset(m, d, qad)
pp.move(m, d, qad, "%s_hover" % col, pp.GRIP_OPEN, 2.0)
print("hover 后 qpos(度): " + " ".join("%+.1f" % np.degrees(v) for v in d.qpos[qad]))
print("            指尖 %s" % np.round(tip(), 4))

pp.move(m, d, qad, kf, pp.GRIP_OPEN, 2.5)
print("grasp 后 qpos(度): " + " ".join("%+.1f" % np.degrees(v) for v in d.qpos[qad]))
print("            指尖 %s" % np.round(tip(), 4))
print("命令     qpos(度): " + " ".join("%+.1f" % np.degrees(v) for v in q_cmd))
print("     关节差(度): " + " ".join("%+.2f" % np.degrees(v)
                                    for v in (d.qpos[qad] - q_cmd)))

# 拿同样的 qpos 做一次纯运动学
q_act = np.array([d.qpos[a] for a in qad])
mujoco.mj_resetData(m, d)
d.qpos[qad] = q_act
mujoco.mj_forward(m, d)
print("\n用实测 qpos 做 FK，指尖 %s" % np.round(tip(), 4))
print("（如果和上面不同，说明是浮点/坐标系问题；如果相同，说明 qpos 确实没到目标）")

mujoco.mj_resetData(m, d)
d.qpos[:] = m.key(kf).qpos
mujoco.mj_forward(m, d)
print("用 keyframe  qpos 做 FK，指尖 %s" % np.round(tip(), 4))

# 看看有没有关节撞限位
print()
for i, j in enumerate(JOINTS):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    lo, hi = m.jnt_range[jid]
    v = d.qpos[qad[i]]
    print("  %-7s %.1f°  范围 [%.1f, %.1f]  余量 %5.1f°"
          % (j, np.degrees(v), np.degrees(lo), np.degrees(hi),
             np.degrees(min(v - lo, hi - v))))
