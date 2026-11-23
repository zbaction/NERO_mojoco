"""腕部深度相机（eye-in-hand）演示与校验：RGB-D 采集 → 点云 → 颜色分割 → 测 3D 位置。

相机装在夹爪（gripper_base）侧面，跟着机械臂运动，所以：
  * 每个姿态只能看到"自己正下方的目标方块 + 附近局部桌面"
  * 校验方式：在每块方块的 hover 姿态各测一次，与真值对比
  * 额外校验：相机是 gripper_base 的子 body、随姿态移动、光轴朝下

    python3 depth_view.py                # 在 green_hover 采一次，出图 + 打表
    python3 depth_view.py --check        # 三个方块各自的 hover 姿态都测一遍（PASS 返回 0）
    python3 depth_view.py --all-poses    # 对 19 个关键帧都取一次，统计"能看到哪些方块"

产物：
    out/depth_rgb.png        彩色图（green_hover）
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

CAM = "wrist_cam"
CUBE_OF = {"red": "cube_red", "green": "cube_green", "blue": "cube_blue"}
RGB_CUBE = {"red": (0.86, 0.13, 0.13), "green": (0.12, 0.68, 0.26),
            "blue": (0.16, 0.34, 0.86)}
# 每个方块在哪个姿态下测最清楚
HOVER_OF = {"red": "red_hover", "green": "green_hover", "blue": "blue_hover"}


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


def colormap(depth, lo=0.10, hi=0.90):
    """turbo 近似：深蓝(远) -> 青 -> 黄 -> 红(近)。腕部相机量程只有 0.5 m 左右。"""
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


def measure(dc, rgb, depth, limit_z=0.75, cols=("red", "green", "blue")):
    """对每个颜色做分割 -> 点云 -> 测 3D 位置。返回 {col: dict}。"""
    pts, pcols, mask = dc.pointcloud(depth, rgb)
    rgb_f = rgb.astype(np.float32) / 255.0
    out = {}
    for col in cols:
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
        # 早期用均值会带来约 +7 mm 的系统性偏移。
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


def goto(m, d, adr, poses, name, grip=0.05):
    mujoco.mj_resetData(m, d)
    d.qpos[adr] = np.radians(np.array(poses[name]))
    d.qpos[7] = d.qpos[8] = grip
    mujoco.mj_forward(m, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all-poses", action="store_true")
    ap.add_argument("--pose", default="green_hover", help="单独采集时用哪个关键帧")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    poses = json.loads(POSES.read_text(encoding="utf-8"))
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)]
           for i in range(1, 8)]
    bid_cam = m.cam_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM)]
    cam_body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(bid_cam))

    dc = DepthCamera(m, d, cam_name=CAM, width=848, height=480, max_depth=1.0)
    fx, fy, cx, cy = dc.intrinsics()
    print("腕部深度相机 %r（挂在 body %r 上）：%dx%d  fovy=%.1f°  fx=fy=%.2f  cx=%.1f  cy=%.1f"
          % (dc.name, cam_body, dc.width, dc.height, dc.fovy, fx, cx, cy))

    goto(m, d, adr, poses, args.pose)
    p0, R = dc.pose()
    print("  %s 姿态下光心世界坐标 [%.3f %.3f %.3f]，光轴 %s"
          % (args.pose, *p0, np.round(-R[:, 2], 3)))

    rgb, depth = dc.capture()
    OUT.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(OUT / "depth_rgb.png")
    Image.fromarray(colormap(depth)).save(OUT / "depth_colormap.png")
    valid = (depth > 0.05) & (depth < dc.max_depth)
    print("  深度图：有效几何 %.1f%%  最近 %.3f m  最远(有效) %.3f m"
          % (100.0 * valid.mean(), depth[valid].min(), depth[valid].max()))

    pts, pcols, _ = dc.pointcloud(depth, rgb)
    write_ply(OUT / "depth_pointcloud.ply", pts, pcols)
    print("  点云：%d 点 -> out/depth_pointcloud.ply" % len(pts))

    ok = True
    if args.check:
        # 每个方块在自己的 hover 姿态下测一次
        print("\n%-6s %-13s %7s %8s  %-24s %-24s %7s %7s"
              % ("颜色", "姿态", "像素", "点数", "测得位置(x,y,z_top)", "真值中心(x,y,z)", "Δxy", "Δz"))
        for col in ("red", "green", "blue"):
            goto(m, d, adr, poses, HOVER_OF[col])
            rgb2, dep2 = dc.capture()
            r = measure(dc, rgb2, dep2, cols=(col,))[col]
            gt, half = ground_truth(m, d)[col]
            if not r.get("ok"):
                ok = False
                print("%-6s %-13s %7d  --   分割失败（像素不足）" % (col, HOVER_OF[col], r["px"]))
                continue
            c = r["center"]
            dxy = float(np.hypot(c[0] - gt[0], c[1] - gt[1]))
            dz = float(r["z_top"] - (gt[2] + half))
            print("%-6s %-13s %7d %8d  [%6.3f %6.3f %6.3f]  [%6.3f %6.3f %6.3f]  %5.1fmm %5.1fmm"
                  % (col, HOVER_OF[col], r["px"], r["n"], *c, *gt, 1000 * dxy, 1000 * dz))
            if dxy > 0.008 or abs(dz) > 0.012:
                ok = False
    else:
        meas = measure(dc, rgb, depth)
        gt = ground_truth(m, d)
        print("\n%-6s %7s %8s  %-24s %-24s %7s %7s"
              % ("颜色", "像素", "点数", "测得位置(x,y,z_top)", "真值中心(x,y,z)", "Δxy", "Δz"))
        for col in ("red", "green", "blue"):
            r = meas[col]
            c_gt, half = gt[col]
            if not r["ok"]:
                print("%-6s %7d    --   （该姿态看不到这一块）" % (col, r["px"]))
                continue
            c = r["center"]
            dxy = float(np.hypot(c[0] - c_gt[0], c[1] - c_gt[1]))
            dz = float(r["z_top"] - (c_gt[2] + half))
            print("%-6s %7d %8d  [%6.3f %6.3f %6.3f]  [%6.3f %6.3f %6.3f]  %5.1fmm %5.1fmm"
                  % (col, r["px"], r["n"], *c, *c_gt, 1000 * dxy, 1000 * dz))

    if args.all_poses:
        print("\n[19 个关键帧的可见性]")
        seen = {}
        for name in poses:
            goto(m, d, adr, poses, name)
            rgb2, dep2 = dc.capture()
            mm = measure(dc, rgb2, dep2)
            vis = [c for c in ("red", "green", "blue") if mm[c]["ok"]]
            seen[name] = vis
            print("  %-16s 可见方块 %s" % (name, ", ".join(vis) if vis else "无"))
        # 每块的 hover 姿态必须能看见自己
        miss = [c for c in ("red", "green", "blue") if c not in seen.get(HOVER_OF[c], [])]
        if miss:
            print("  [FAIL] 这些方块在自己的 hover 姿态下不可见：%s" % ", ".join(miss))
            ok = False
        else:
            print("  三个方块在各自 hover 姿态下均可见")

    print("\n结果: %s" % ("PASS" if ok else "FAIL"))
    dc.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
