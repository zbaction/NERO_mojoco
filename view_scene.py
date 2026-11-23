#!/usr/bin/env python3
"""打开 NERO 桌面场景：交互窗口 + 演示抓取。

用法：
    python3 view_scene.py             # 交互窗口，从 ready 姿态开始
    python3 view_scene.py --demo      # 交互窗口，先自动演示一遍抓取
    python3 view_scene.py --check     # 只做无窗自检（CI 友好）
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import mujoco
import mujoco.viewer

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "nero_table_scene.xml"
ARM = ["joint%d" % i for i in range(1, 8)]
GRIP_OPEN, GRIP_CLOSE = 0.050, 0.0


def load():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    qad = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in ARM]
    return m, d, qad


def drive(m, d, qad, key, grip, secs, viewer=None):
    """把关节伺服到关键帧，同时同步交互窗口。"""
    q = np.array([m.key(key).qpos[x] for x in qad])
    d.ctrl[:7] = q
    d.ctrl[7] = d.ctrl[8] = grip
    n = int(secs / m.opt.timestep)
    for i in range(n):
        mujoco.mj_step(m, d)
        if viewer is not None and i % 10 == 0:
            viewer.sync()


def demo(m, d, qad, viewer=None):
    drive(m, d, qad, "hover", GRIP_OPEN, 2.0, viewer)
    drive(m, d, qad, "grasp", GRIP_OPEN, 2.5, viewer)
    drive(m, d, qad, "grasp", GRIP_CLOSE, 2.0, viewer)
    drive(m, d, qad, "hover", GRIP_CLOSE, 2.5, viewer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="先自动演示一遍抓取")
    ap.add_argument("--check", action="store_true", help="无窗自检")
    a = ap.parse_args()

    m, d, qad = load()
    d.qpos[:] = m.key("ready").qpos
    d.ctrl[:7] = [m.key("ready").qpos[x] for x in qad]
    d.ctrl[7] = d.ctrl[8] = GRIP_OPEN
    mujoco.mj_forward(m, d)

    print("模型：%d 个 body / %d 个 geom / %d 个关节 / %d 个执行器"
          % (m.nbody, m.ngeom, m.njnt, m.nu))

    def bmass(n):
        return float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)])

    arm = ["nero_base"] + ["link%d" % i for i in range(1, 8)] + ["end_effector"]
    grip = ["gripper_base", "finger_l", "finger_r"]
    detail = ", ".join("%s=%.3f" % (n, bmass(n)) for n in arm + grip)
    print("本体（含法兰）%.3f kg + 夹爪 %.3f kg"
          % (sum(bmass(n) for n in arm), sum(bmass(n) for n in grip)))
    print("明细：%s" % detail)

    if a.check:
        demo(m, d, qad)
        green = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_green")
        print("自检：抓取后绿方块 z = %.4f m" % d.xpos[green][2])
        return 0 if d.xpos[green][2] > 0.85 else 1

    with mujoco.viewer.launch_passive(m, d) as v:
        if a.demo:
            demo(m, d, qad, v)
        while v.is_running():
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(m.opt.timestep)
    return 0


if __name__ == "__main__":
    import os
    import sys
    import traceback

    rc = 0
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc()
        rc = 1
    finally:
        # os._exit 不冲刷缓冲，必须手动 flush，否则管道里看不到输出
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc if isinstance(rc, int) else 0)   # 规避 WSLg 退出时 GLFW 的 segfault
