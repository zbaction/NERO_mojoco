#!/usr/bin/env python3
"""腕部相机（eye-in-hand）安装示意 + 第一视角：一张图说明"相机装在哪、看到什么"。

左图：外部相机拍 gripper_base 特写（相机壳体、镜头支架、夹爪、目标方块）
右图：wrist_cam 自己渲染的画面（同一姿态）

自动取景用解析投影（包围球角点 -> 像素），不做渲染差分迭代。

用法：
    python3 render_depthcam_shot.py            # 出 out/wrist_cam_mount.png
    python3 render_depthcam_shot.py --where    # 只打印几何关系
    python3 render_depthcam_shot.py --pose green_grasp
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "nero_table_scene.xml"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
W, H = 848, 480

# PIL 默认位图字体没有中文字形，标题会整条画成空白。
_CJK = "/home/zhangb/.local/share/fonts/NotoSansCJKsc-Regular.otf"


def _font(size):
    if pathlib.Path(_CJK).exists():
        try:
            return ImageFont.truetype(_CJK, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_CAP = _font(20)


def bid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def cam_basis(az_deg, el_deg):
    """MuJoCo free 相机的朝向（源码约定）：fwd 的 z 分量取正号。"""
    az, el = np.radians(az_deg), np.radians(el_deg)
    fwd = np.array([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])
    up_w = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(fwd, up_w))) > 0.999:
        up_w = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up_w)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def project(pts, right, up, fwd, look, dist, fovy, w, h):
    """世界点 -> 像素坐标（允许落在画面外，便于算边距）。"""
    pos = look - dist * fwd
    v = pts - pos
    zc = v @ fwd
    xc = v @ right
    yc = v @ up
    fy = 0.5 * h / np.tan(np.radians(fovy) / 2)
    u = w / 2 + xc / np.maximum(zc, 1e-6) * fy
    vv = h / 2 - yc / np.maximum(zc, 1e-6) * fy
    return u, vv, zc


def group_points(m, d, body_names, geom_prefixes=()):
    """指定 body（及其子 body）下所有视觉 geom 的包围球角点（世界坐标）。

    注意：`wrist_cam_body` / `wrist_cam_lens` / `wrist_cam_mount` 是 **geom** 名，
    不是 body 名（`mj_name2id(mjOBJ_BODY, ...)` 返回 -1），所以要用名字前缀筛 geom。
    """
    ids = set()
    for n in body_names:
        i = bid(m, n)
        if i >= 0:
            ids.add(i)
    pts = []
    for g in range(m.ngeom):
        gname = m.geom(g).name
        hit = int(m.geom_bodyid[g]) in ids
        if not hit and geom_prefixes:
            hit = any(gname.startswith(p) for p in geom_prefixes)
        if not hit:
            continue
        c = d.geom_xpos[g]
        # geom_rbound 是**包围球**半径（box 的对角线），直接拿来当立方体角点会
        # 把包围盒放大 √3 倍 → 自动取景算出的 dist 偏大一倍。乘 0.6 近似贴合真实外形。
        r = float(m.geom_rbound[g]) * 0.60
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(c + r * np.array([sx, sy, sz]))
    return np.array(pts)


def frame_group(m, d, body_names, geom_prefixes=(), fovy=34.0, fill=0.82,
                az_range=(-40, 70)):
    pts = group_points(m, d, body_names, geom_prefixes)
    if len(pts) == 0:
        raise SystemExit("找不到 %s 的几何体" % body_names)
    look = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    best = None
    for az in range(az_range[0], az_range[1] + 1, 6):
        for el in (-24, -16, -8, 0, 8, 16, 24, 32):
            right, up, fwd = cam_basis(az, el)
            lo, hi = 0.05, 20.0
            for _ in range(46):
                mid = 0.5 * (lo + hi)
                u, v, z = project(pts, right, up, fwd, look, mid, fovy, W, H)
                if (z <= 0.02).any():
                    lo = mid
                    continue
                frac = max((u.max() - u.min()) / W, (v.max() - v.min()) / H)
                if frac > fill:
                    lo = mid
                else:
                    hi = mid
            dist = 0.5 * (lo + hi)
            u, v, z = project(pts, right, up, fwd, look, dist, fovy, W, H)
            if (z <= 0.02).any():
                continue
            frac = max((u.max() - u.min()) / W, (v.max() - v.min()) / H)
            margin = min(u.min(), v.min(), W - 1 - u.max(), H - 1 - v.max())
            score = margin - 400.0 * abs(frac - fill)
            if best is None or score > best[0]:
                best = (score, az, el, dist, look.copy(), frac, margin)
    return best


def render_free(m, d, az, el, dist, look, fovy):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    m.vis.global_.fovy = fovy
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, cam)
    img = r.render().copy()
    r.close()
    return img


def render_cam(m, d, cam_name):
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera=cid)
    img = r.render().copy()
    r.close()
    return img


def caption(im, text):
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 32], fill=(0, 0, 0))
    dr.text((8, 5), text, fill=(255, 235, 120), font=F_CAP)
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="green_hover")
    ap.add_argument("--where", action="store_true")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    poses = json.loads((HERE / "poses.json").read_text(encoding="utf-8"))
    if args.pose not in poses:
        raise SystemExit("未知位姿 %s" % args.pose)
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)]
           for i in range(1, 8)]
    mujoco.mj_resetData(m, d)
    d.qpos[adr] = np.radians(np.array(poses[args.pose]))
    d.qpos[7] = d.qpos[8] = 0.050
    mujoco.mj_forward(m, d)

    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    cb = int(m.cam_bodyid[cid])
    p = d.cam_xpos[cid]
    R = np.array(d.cam_xmat[cid]).reshape(3, 3)
    ax = -R[:, 2]
    grip = bid(m, "gripper_base")
    pg = d.xpos[grip]
    Rg = np.array(d.xmat[grip]).reshape(3, 3)
    local = Rg.T @ (p - pg)

    print("姿态 %s" % args.pose)
    print("相机挂在 body '%s'（id %d）" % (m.body(cb).name, cb))
    print("光心世界坐标 %s   光轴 %s" % (np.round(p, 4), np.round(ax, 3)))
    print("在 gripper_base 系里的位置 %s  |p|=%.1f mm" % (np.round(local, 4),
                                                        1000 * np.linalg.norm(local)))
    print("相机到桌面 %.3f m，到绿色方块 %.3f m" % (
        p[2] - 0.740,
        np.linalg.norm(p - d.xpos[bid(m, "cube_green")])))
    if args.where:
        return 0

    bodies = ["link6", "link7", "end_effector", "gripper_base", "finger_l", "finger_r"]
    score, az, el, dist, look, frac, margin = frame_group(
        m, d, bodies, geom_prefixes=("wrist_cam_",), fovy=40.0, fill=0.86,
        az_range=(-180, 180))
    print("特写取景 az=%d el=%d dist=%.3f 占比 %.2f 边距 %.0f px"
          % (az, el, dist, frac, margin))

    left = render_free(m, d, az, el, dist, look, 40.0)
    right = render_cam(m, d, "wrist_cam")

    # 外部特写用的是 free 相机，headlight 只照亮正对方向，腕部背光偏暗。
    # 这里做一次固定的曝光提升（线性 2.0× + 12），并在标题里注明，不伪装原始亮度。
    lg = np.clip(np.asarray(left, dtype=np.float32) * 2.0 + 12.0, 0, 255).astype(np.uint8)
    left = caption(Image.fromarray(lg), "gripper_base 特写（亮度 2.0x，相机装在掌心 +x 侧）")
    right = caption(Image.fromarray(right), "wrist_cam 第一视角（%s）" % args.pose)
    out = Image.new("RGB", (W * 2 + 8, H), (16, 16, 18))
    out.paste(left, (0, 0))
    out.paste(right, (W + 8, 0))
    out.save(OUT / "wrist_cam_mount.png")

    a = np.asarray(left).astype(float)
    b = np.asarray(right).astype(float)
    print("左图 mean %.1f std %.1f ｜ 右图 mean %.1f std %.1f"
          % (a.mean(), a.std(), b.mean(), b.std()))
    print("写出 out/wrist_cam_mount.png（%dx%d）" % out.size)

    ok = left is not None and b.std() > 8.0
    print("结果: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
