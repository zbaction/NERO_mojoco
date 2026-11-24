#!/usr/bin/env python3
"""出成果图：NERO 抓起绿方块。相机按"机械臂实际像素包围盒"迭代自动取景。

用法：
    python3 render_showcase.py                # 出图到 out/
    python3 render_showcase.py --viewer       # 交互窗口
    python3 render_showcase.py --pose ready   # 换关键帧
    python3 render_showcase.py --fill 0.75    # 主体占画面高度的比例
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import mujoco
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "nero_table_scene.xml"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

ARM = ["joint%d" % i for i in range(1, 8)]
GRIP_OPEN, GRIP_CLOSE = 0.050, 0.0
W, H = 1280, 720


def grab_and_lift(m, d, qad):
    """走到 grasp 夹住绿块，再竖直提回 hover。"""
    def drive(key, grip, secs):
        q = np.array([m.key(key).qpos[x] for x in qad])
        d.ctrl[:7] = q
        d.ctrl[7] = d.ctrl[8] = grip
        for _ in range(int(secs / m.opt.timestep)):
            mujoco.mj_step(m, d)

    d.qpos[:] = m.key("ready").qpos
    d.ctrl[:7] = [m.key("ready").qpos[x] for x in qad]
    d.ctrl[7] = d.ctrl[8] = GRIP_OPEN
    mujoco.mj_forward(m, d)
    drive("green_hover", GRIP_OPEN, 2.0)
    drive("green_grasp", GRIP_OPEN, 2.5)
    drive("green_grasp", GRIP_CLOSE, 2.0)
    drive("green_hover", GRIP_CLOSE, 2.5)


def arm_mask(r, d, cam):
    """两次渲染差分，得到只含机械臂（geomgroup 2）的掩膜。"""
    opt = mujoco.MjvOption()
    r.update_scene(d, cam, opt)
    with_arm = r.render().astype(int)
    opt.geomgroup[2] = 0
    r.update_scene(d, cam, opt)
    without = r.render().astype(int)
    return np.abs(with_arm - without).sum(axis=2) > 12


def arm_points(m, d):
    """机械臂（geomgroup 2）所有几何体的世界坐标采样点，用于解析投影。"""
    root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "nero_base")]
    pts = []
    for g in range(m.ngeom):
        if m.geom_group[g] != 2:
            continue
        if m.body_rootid[m.geom_bodyid[g]] != root:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a = m.mesh_vertadr[mid]
            n = m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n]
            pts.append(v @ R.T + c)
        else:
            r = float(m.geom_rbound[g])
            off = np.array([[sx, sy, sz] for sx in (-r, r)
                            for sy in (-r, r) for sz in (-r, r)])
            pts.append(off + c)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def cam_basis(cam):
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    off = np.array([math.cos(el) * math.cos(az),
                    math.cos(el) * math.sin(az),
                    math.sin(el)])
    f = -off
    right = np.cross(f, np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else right / n
    up = np.cross(right, f)
    return off, f, right, up


def project(pts, lookat, dist, off, f, right, up, fovy, aspect, w, h):
    """世界点 -> 像素坐标（不做裁剪，允许超出画面）。"""
    pos = lookat + dist * off
    rel = pts - pos
    zc = rel @ f
    xc = rel @ right
    yc = rel @ up
    zc = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
    th = math.tan(fovy / 2)
    u = (xc / zc) / (th * aspect)
    v = (yc / zc) / th
    px = (u + 1) * 0.5 * (w - 1)
    py = (1 - v) * 0.5 * (h - 1)
    return px, py


def autoframe(m, d, cam, pts, fill=0.80, iters=25, verbose=False):
    """解析自动取景：把机械臂点云投影到画面中心并填充到 fill 比例。

    全程用解析投影，不做渲染差分，所以主体即便暂时超出画面也不会被裁剪，
    迭代是单调收敛的（带阻尼，避免过冲）。
    """
    fovy = math.radians(m.vis.global_.fovy)
    aspect = W / H
    lookat = np.asarray(cam.lookat, float).copy()
    dist = float(cam.distance)
    for it in range(iters):
        off, f, right, up = cam_basis(cam)
        px, py = project(pts, lookat, dist, off, f, right, up,
                         fovy, aspect, W, H)
        x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
        wpx, hpx = x1 - x0, y1 - y0
        frac = max(wpx / W, hpx / H)
        du = 0.5 * (x0 + x1) - 0.5 * (W - 1)
        dv = 0.5 * (y0 + y1) - 0.5 * (H - 1)
        if verbose:
            print("    第 %2d 轮：bbox %.0fx%.0f px 占比 %.0f%%x%.0f%%  "
                  "偏移(%+.0f,%+.0f)  dist=%.2f"
                  % (it, wpx, hpx, 100 * wpx / W, 100 * hpx / H, du, dv, dist))
        if abs(frac - fill) < 0.02 and max(abs(du), abs(dv)) < 4:
            break
        per_px = 2 * dist * math.tan(fovy / 2) / H
        lookat = lookat + (0.6 * du * right - 0.6 * dv * up) * per_px
        if frac > 1e-6:
            dist *= (frac / fill) ** 0.7
    cam.lookat[:] = lookat
    cam.distance = dist
    return cam


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--pose", default="lift", choices=["lift", "ready"])
    ap.add_argument("--fill", type=float, default=0.80)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    qad = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in ARM]
    green = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_green")

    if a.pose == "lift":
        grab_and_lift(m, d, qad)
        print("抓取结果：方块 z = %.4f m（桌面上表面 0.740）" % d.xpos[green][2])
    else:
        d.qpos[:] = m.key("ready").qpos
        d.ctrl[:7] = [m.key("ready").qpos[x] for x in qad]
        d.ctrl[7] = d.ctrl[8] = GRIP_OPEN
        mujoco.mj_forward(m, d)

    if a.viewer:
        import time
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():
                v.sync()
                time.sleep(0.01)
        return 0

    r = mujoco.Renderer(m, H, W)
    m.vis.global_.fovy = 38
    pts = arm_points(m, d)
    print("机械臂取景点云 %d 个点" % len(pts))

    shots = [
        ("showcase_grasp", 38, -12),
        ("showcase_front", 2, -4),
        ("showcase_side", 92, -6),
        ("showcase_iso", 138, -18),
    ]
    for name, az, el in shots:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (0.02, 0.0, 0.95)
        cam.distance = 1.2
        cam.azimuth = az
        cam.elevation = el
        if a.verbose:
            print("  %s:" % name)
        cam = autoframe(m, d, cam, pts, fill=a.fill, verbose=a.verbose)
        r.update_scene(d, cam)
        img = r.render()
        Image.fromarray(img).save(OUT / (name + ".png"))
        print("  写出 %-16s dist=%.2f m lookat=%s"
              % (name, cam.distance, np.round(cam.lookat, 3)))
    r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
