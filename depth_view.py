"""深度相机演示与校验：RGB-D 采集 → 点云 → 颜色分割 → 测出三块方块的 3D 位置。

    python3 depth_view.py                # 采集一次，出图 + 打表
    python3 depth_view.py --check        # 同上，并把测量误差当断言（PASS 返回 0）
    python3 depth_view.py --all-poses    # 对 19 个关键帧都取一次，统计视野与遮挡

产物：
    out/depth_rgb.png        彩色图
    out/depth_colormap.png   深度伪彩图（turbo，越近越红）
    out/depth_pointcloud.ply 世界系点云（可用 MeshLab / CloudCompare 打开）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import mujoco
import PIL.Image as Image

from depth_cam import DepthCamera

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "nero_table_scene.xml"
OUT = HERE / "out"
POSES = HERE / "poses.json"

CUBE_OF = {"red": "cube_red", "green": "cube_green", "blue": "cube_blue"}
RGB_CUBE = {"red": (0.86, 0.13, 0.13), "green": (0.12, 0.68, 0.26),
            "blue": (0.16, 0.34, 0.86)}


def rgb_to_hsv(rgb):
    """rgb: float [0,1]，shape (...,3) -> h in [0,1), s, v。"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    idx = nz & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / d[idx]) % 6.0
    idx = nz & (mx == g)
    h[idx] = (b[idx] - r[idx]) / d[idx] + 2.0
    idx = nz & (mx == b)
    h[idx] = (r[idx] - g[idx]) / d[idx] + 4.0
    h = h / 6.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


def color_mask(rgb, name):
    h, s, v = rgb_to_hsv(rgb)
    if name == "red":
        hue = ((h < 0.055) | (h > 0.945)) & (s > 0.45) & (v > 0.12)
    elif name == "green":
        hue = (h > 0.20) & (h < 0.46) & (s > 0.32) & (v > 0.08)
    else:
        hue = (h > 0.50) & (h < 0.75) & (s > 0.30) & (v > 0.08)
    return hue


def colormap(depth, lo=0.3, hi=2.4):
    """turbo 近似：深蓝(远) -> 青 -> 黄 -> 红(近)。"""
    t = np.clip((hi - depth) / (hi - lo), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def write_ply(path, pts, cols):
    n = len(pts)
    with open(path, "w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write("element vertex %d\n" % n)
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for p, c in zip(pts, cols):
            fh.write("%.5f %.5f %.5f %d %d %d\n" % (p[0], p[1], p[2], c[0], c[1], c[2]))


def ground_truth(m, d):
    """从模型里读三块方块的真实位姿（不是硬编码）。"""
    gt = {}
    for col, body in CUBE_OF.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, body + "_geom")
        gt[col] = (np.array(d.xpos[bid]), float(m.geom_size[gid][2]))
    return gt


def measure(dc, rgb, depth, limit_z=0.75):
    """对每个颜色做分割 -> 点云 -> 测 3D 位置。返回 {col: dict}。"""
    pts, cols, mask = dc.pointcloud(depth, rgb)
    rgb_f = rgb.astype(np.float32) / 255.0
    out = {}
    for col in ("red", "green", "blue"):
        cm = color_mask(rgb_f, col) & mask
        n = int(cm.sum())
        if n < 120:
            out[col] = {"px": n, "ok": False}
            continue
        p = dc.unproject(*np.nonzero(cm)[::-1], depth[cm])
        # 只要桌面以上的点（排除桌面对应像素被误判成方块色）
        keep = p[:, 2] > limit_z
        p = p[keep]
        if len(p) < 80:
            out[col] = {"px": n, "ok": False}
            continue
        z_top = float(np.percentile(p[:, 2], 97))
        # 顶面选点：只要顶面附近 3 mm。注意这里**不能只取均值** ——
        # 立方体侧面朝相机的那两面会有一条同高度的窄带被一起选进来，
        # 它们全在顶面投影范围内（只影响均值，不影响外接框），
        # 早期用均值会带来约 +7 mm 的系统性偏移（实测红/绿/蓝都是 +x+y 方向）。
        # 用 x/y 的分位中点（bbox 中心）对这条窄带免疫。
        top = p[p[:, 2] > z_top - 0.003]
        lo = np.percentile(top[:, :2], 1.0, axis=0)
        hi = np.percentile(top[:, :2], 99.0, axis=0)
        center_xy = 0.5 * (lo + hi)
        out[col] = {"px": n, "ok": True, "n": len(p), "n_top": len(top),
                    "center": np.array([center_xy[0], center_xy[1], z_top]),
                    "z_top": z_top,
                    "bbox_lo": p.min(axis=0), "bbox_hi": p.max(axis=0)}
    return out


def scene_option_no_sky():
    opt = mujoco.MjvOption()
    return opt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all-poses", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.key("ready").qpos
    mujoco.mj_forward(m, d)

    dc = DepthCamera(m, d)
    fx, fy, cx, cy = dc.intrinsics()
    print("深度相机 %r：%dx%d  fovy=%.1f°  fx=fy=%.2f  cx=%.1f  cy=%.1f"
          % (dc.name, dc.width, dc.height, dc.fovy, fx, cx, cy))
    p0, R = dc.pose()
    print("光心世界坐标 [%.3f %.3f %.3f]，光轴 %s"
          % (*p0, np.round(-R[:, 2], 3)))

    rgb, depth = dc.capture()
    OUT.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(OUT / "depth_rgb.png")
    Image.fromarray(colormap(depth)).save(OUT / "depth_colormap.png")
    valid = (depth > 0.05) & (depth < dc.max_depth)
    print("深度图：有效几何 %.1f%%  最近 %.3f m  最远(有效) %.3f m"
          % (100.0 * valid.mean(), depth[valid].min(), depth[valid].max()))

    pts, cols, _ = dc.pointcloud(depth, rgb)
    write_ply(OUT / "depth_pointcloud.ply", pts, cols)
    print("点云：%d 点 -> out/depth_pointcloud.ply" % len(pts))

    meas = measure(dc, rgb, depth)
    gt = ground_truth(m, d)
    print("\n%-6s %7s %8s  %-26s %-26s %6s %6s"
          % ("颜色", "像素", "点数", "测得中心(x,y,z)", "真实中心(x,y,z)", "Δxy", "Δz"))
    ok = True
    for col in ("red", "green", "blue"):
        r = meas[col]
        c_gt, half = gt[col]
        if not r["ok"]:
            ok = False
            print("%-6s %7d   --   分割失败（像素不足）" % (col, r["px"]))
            continue
        c = r["center"]
        dxy = float(np.hypot(c[0] - c_gt[0], c[1] - c_gt[1]))
        dz = float(r["z_top"] - (c_gt[2] + half))
        print("%-6s %7d %8d  [%6.3f %6.3f %6.3f]   [%6.3f %6.3f %6.3f]  %5.1fmm %5.1fmm"
              % (col, r["px"], r["n"], *c, *c_gt, 1000 * dxy, 1000 * dz))
        if dxy > 0.006 or abs(dz) > 0.008:
            ok = False

    # 桌面平面拟合（用篮子附近以外的桌面点，验证深度标定）
    flat = pts[np.abs(pts[:, 2] - 0.74) < 0.02]
    if len(flat) > 500:
        print("\n桌面点 %d 个，z 均值 %.4f m（真实 0.7400），标准差 %.4f m"
              % (len(flat), flat[:, 2].mean(), flat[:, 2].std()))

    if args.all_poses:
        poses = json.loads(POSES.read_text(encoding="utf-8"))
        adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)]
               for i in range(1, 8)]
        print("\n[19 个关键帧的可见性]")
        bad = []
        for name in poses:
            mujoco.mj_resetData(m, d)
            d.qpos[adr] = np.radians(np.array(poses[name]))
            d.qpos[7] = d.qpos[8] = 0.05
            mujoco.mj_forward(m, d)
            rgb2, dep2 = dc.capture()
            mm = measure(dc, rgb2, dep2)
            vis = [c for c in ("red", "green", "blue") if mm[c]["ok"]]
            print("  %-16s 可见方块 %s" % (name, ",".join(vis) if vis else "无"))
            if name.endswith("_hover") or name == "ready":
                if len(vis) < 3:
                    bad.append(name)
        if bad:
            print("  [WARN] 这些姿态下没能同时看到三块方块：%s" % ", ".join(bad))

    print("\n结果: %s" % ("PASS" if ok else "FAIL"))
    dc.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
