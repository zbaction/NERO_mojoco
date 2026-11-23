#!/usr/bin/env python3
"""验证交互窗口能开起来并保持运行（无人工干预）。"""
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path(os.path.expanduser("~/NERO/nero_table_scene.xml"))
d = mujoco.MjData(m)
d.qpos[:] = m.key("ready").qpos
mujoco.mj_forward(m, d)

print("model=%s  nbody=%d" % (m.names[:40].split(b"\x00")[0].decode(errors="ignore"), m.nbody))
t0 = time.time()
with mujoco.viewer.launch_passive(m, d) as v:
    v.sync()
    print("viewer 已启动，is_running=%s" % v.is_running())
    for i in range(150):
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(0.02)
    print("跑完 %.1f s，is_running=%s" % (time.time() - t0, v.is_running()))
print("正常退出")
os._exit(0)
