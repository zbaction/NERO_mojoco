#!/usr/bin/env python3
"""核对：同一组关节角，用 mj_forward 求 pad 中点，确认与 validate_nero.py 一致。"""
from __future__ import annotations

import pathlib

import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
m = mujoco.MjModel.from_xml_path(str(HERE / "nero_table_scene.xml"))
d = mujoco.MjData(m)
ARM = ["joint%d" % i for i in range(1, 8)]
QAD = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                for j in ARM])
PADL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
PADR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")
print("QAD =", QAD)
print("gripper_base body id =",
      mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"),
      "finger_l =", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "finger_l"))
print("pad 局部 pos: L=%s R=%s" % (m.geom_pos[PADL], m.geom_pos[PADR]))

for name in ("ready", "hover", "grasp"):
    d.qpos[:] = m.key(name).qpos
    mujoco.mj_forward(m, d)
    mid = 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])
    print("%-6s q=%s" % (name, " ".join("%+.2f" % x for x in np.degrees(d.qpos[QAD]))))
    print("       padL %s padR %s mid %s"
          % (np.round(d.geom_xpos[PADL], 4), np.round(d.geom_xpos[PADR], 4),
             np.round(mid, 4)))

# 只设置 7 个关节，其余保持 ready 的 qpos
d.qpos[:] = m.key("ready").qpos
d.qpos[QAD] = m.key("grasp").qpos[QAD]
mujoco.mj_forward(m, d)
mid = 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])
print("only-arm-subset grasp mid %s" % np.round(mid, 4))
