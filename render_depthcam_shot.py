#!/usr/bin/env python3
"""出"深度相机安装位置"示意图：从侧后方看整个场景（桌子 + 机械臂 + 三脚架相机）。

用解析投影自动取景：把场景里所有 geom 的包围球角点投影到画面，
带阻尼地调整 lookat / distance，使联合包围盒占画面 82% 且居中。

用法：
    python3 render_depthcam_shot.py          # 出 out/depth_cam_scene.png
    python3 render_depthcam_shot.py --where  # 只打印几何关系
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import mujoco
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "nero_table_scene.xml"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
W, H = 1280, 720


def bid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def scene_points(m, d, max_rbound=2.0):
    """所有 geom 的包围球 ± 半径的 8 个角点（世界坐标）。

    跳过 rbound 过大的 geom（地面是无限平面，会把自动取景撑爆）。
    """
    pts = []
    for g in range(m.ngeom):
        r = float(m.geom_rbound[g])
        if r > max_rbound:
            continue
        c = d.geom_xpos[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(c + r * np.array([sx, sy, sz]))
    return np.array(pts)


def project(pts, cam, fovy, w, h):
    """把世界点投到像素坐标（允许超出画面）。

    cam = {"lookat": (3,), "R": 3x3, "dist": float}
    R 的列是相机的 right / up / backward 轴，所以视线方向是 -R[:,2]。
    相机位置 = lookat - dist * 视线方向。
    """
    R = cam["R"]
    fwd = -R[:, 2]
    pos = np.asarray(cam["lookat"]) - cam["dist"] * fwd
    p = pts - pos
    z = p @ fwd
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    # MuJoCo 的 fovy 是**纵向**视场：fy = (h/2)/tan(fovy/2)，方形像素 → fx = fy
    fy = (h / 2) / np.tan(np.radians(fovy) / 2)
    u = (p @ R[:, 0]) / z * fy + w / 2
    v = -(p @ R[:, 1]) / z * fy + h / 2
    return u, v, z


def cam_basis(az_deg, el_deg):
    """复刻 MuJoCo 自由相机约定（mjv_updateCamera）：

        pos = lookat + dist * (cos(el)cos(az), cos(el)sin(az), -sin(el))
        fwd = -(...)          # 从 pos 指向 lookat

    注意 z 分量取 **负号**：elevation 为负表示相机在 lookat 上方俯视。
    """
    az, el = np.radians(az_deg), np.radians(el_deg)
    fwd = np.array([-np.cos(el) * np.cos(az),
                    -np.cos(el) * np.sin(az),
                    np.sin(el)])
    fwd /= np.linalg.norm(fwd)
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.column_stack([right, up, -fwd])


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    d.qpos[:] = m.key("ready").qpos
    mujoco.mj_forward(m, d)

    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
    p = d.cam_xpos[cid]
    R0 = np.array(d.cam_xmat[cid]).reshape(3, 3)
    print("深度相机光心 [%.2f %.2f %.2f]  光轴 %s" % (*p, np.round(-R0[:, 2], 3)))
    print("云台 body 世界坐标 %s" % np.round(d.xpos[bid(m, "depth_cam_head")], 3))
    print("桌面范围 x[-0.40,0.40] y[-0.60,0.60] z=0.74")
    print("相机在桌面外：%s" % ("是" if (abs(p[0]) > 0.40 or abs(p[1]) > 0.60) else "否"))
    if "--where" in sys.argv:
        return 0

    pts = scene_points(m, d)
    fovy = 44.0
    fill = 0.80

    def frame(az, el):
        """固定朝向，lookat 取场景包围盒中心，对 dist 二分使主体占画面 fill。"""
        R = cam_basis(az, el)
        look = 0.5 * (pts.min(axis=0) + pts.max(axis=0))

        def frac_at(dist):
            u, v, z = project(pts, {"lookat": look, "R": R, "dist": dist},
                              fovy, W, H)
            if (z <= 0.05).any():
                return 99.0
            return max((u.max() - u.min()) / W, (v.max() - v.min()) / H)

        lo, hi = 0.3, 20.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if frac_at(mid) > fill:
                lo = mid
            else:
                hi = mid
        dist = 0.5 * (lo + hi)
        u, v, z = project(pts, {"lookat": look, "R": R, "dist": dist},
                          fovy, W, H)
        frac = max((u.max() - u.min()) / W, (v.max() - v.min()) / H)
        margin = min(u.min(), v.min(), W - 1 - u.max(), H - 1 - v.max())
        return R, look, dist, frac, margin

    best = None
    for az in (300, 310, 315, 320, 330):
        for el in (-26, -30, -34):
            R, look, dist, frac, margin = frame(az, el)
            score = margin - 200 * abs(frac - fill)
            if best is None or score > best[0]:
                best = (score, az, el, dist, look.copy(), frac, margin)

    score, az, el, dist, look, frac, margin = best
    print("取景：az=%d el=%d dist=%.2f lookat=%s 占比 %.2f 边距 %.0f px"
          % (az, el, dist, np.round(look, 3), frac, margin))

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
    Image.fromarray(img).save(OUT / "depth_cam_scene.png")
    print("写出 out/depth_cam_scene.png（%dx%d）" % (W, H))

    # 客观确认三脚架真的画出来了
    r2 = mujoco.Renderer(m, H, W)
    r2.update_scene(d, cam)
    a = r2.render().copy().astype(int)
    saved = {}
    for nm in ("depth_cam_head", "depth_cam_stand"):
        i = bid(m, nm)
        saved[i] = m.body_pos[i].copy()
        m.body_pos[i][2] -= 50.0
    mujoco.mj_forward(m, d)
    r2.update_scene(d, cam)
    b = r2.render().copy().astype(int)
    for i, v in saved.items():
        m.body_pos[i][:] = v
    r2.close()
    diff = np.abs(a - b).sum(axis=2) > 12
    print("三脚架+云台占画面 %.2f%%（%d px）" % (100 * diff.mean(), diff.sum()))
    return 0 if diff.sum() > 3000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
