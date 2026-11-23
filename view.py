#!/usr/bin/env python3
"""在 MuJoCo 里打开「一张桌子 + 桌面上三个不同颜色的方块」。

用法:
    python view.py            # 打开交互窗口（WSL 走 WSLg 显示到 Windows 桌面）
    python view.py --render   # 不开窗口，离屏渲染 png 到 ./out/
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
XML = HERE / "desk_three_cubes.xml"
CUBES = ("cube_red", "cube_green", "cube_blue")


def load():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def summary(model, data):
    print("mujoco {}".format(mujoco.__version__))
    print("模型: {}".format(XML.name))
    print("  nbody={} ngeom={} nq={} nv={}".format(model.nbody, model.ngeom, model.nq, model.nv))
    for name in CUBES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name + "_geom")
        mid = model.geom_matid[gid]
        rgba = model.mat_rgba[mid] if mid >= 0 else model.geom_rgba[gid]
        print("  {:<11} pos={} color={}".format(
            name, np.round(data.xpos[bid], 3), np.round(rgba[:3], 2)))


def run_viewer():
    import mujoco.viewer

    model, data = load()
    summary(model, data)
    print("\n窗口已打开：鼠标左键旋转 / 右键平移 / 滚轮缩放，关掉窗口退出。")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            viewer.cam.distance = 2.4
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
    finally:
        # GLFW/WSLg 在解释器正常退出时会 segfault，直接硬退出
        sys.stdout.flush()
        os._exit(0)


def run_render(out_dir):
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    model, data = load()
    summary(model, data)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    for cam_name in ("overview", "front", "top"):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        renderer.update_scene(data, camera=cam)
        path = out_dir / "desk_{}.png".format(cam_name)
        Image.fromarray(renderer.render()).save(path)
        print("  写出 {}".format(path))
    renderer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="只离屏渲染 png，不打开窗口")
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    if not XML.exists():
        print("找不到模型文件: {}".format(XML), file=sys.stderr)
        return 1
    if args.render:
        run_render(pathlib.Path(args.out))
    else:
        run_viewer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())