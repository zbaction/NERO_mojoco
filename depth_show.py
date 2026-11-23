"""深度相机演示图：把 RGB / 深度伪彩 / 点云三视图 + 检测框拼成一张成果图。

    python3 depth_show.py            # -> out/depth_camera.png（2x2 拼版）
    python3 depth_show.py --multi    # 额外出 19 姿态里的 3 个代表姿态
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import mujoco
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from depth_cam import DepthCamera
from depth_view import color_mask, colormap, measure, ground_truth, write_ply

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
SCENE = HERE / "nero_table_scene.xml"
POSE_NAMES = ("ready", "green_hover", "green_grasp")
CUBE_COL = {"red": (230, 40, 40), "green": (40, 200, 90), "blue": (60, 110, 230)}

FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def font(size=20):
    for p in FONT_PATHS:
        if pathlib.Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_cross(draw, uv, color, r=14, w=3):
    u, v = uv
    draw.line([u - r, v, u + r, v], fill=color, width=w)
    draw.line([u, v - r, u, v + r], fill=color, width=w)
    draw.ellipse([u - r, v - r, u + r, v + r], outline=color, width=w)


def render_pose(m, d, poses, name):
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)]
           for i in range(1, 8)]
    mujoco.mj_resetData(m, d)
    d.qpos[adr] = np.radians(np.array(poses[name]))
    d.qpos[7] = d.qpos[8] = 0.05
    mujoco.mj_forward(m, d)


def annotate(rgb, dc, meas, gt, title):
    img = Image.fromarray(rgb).copy()
    dr = ImageDraw.Draw(img)
    f = font(20)
    fs = font(16)
    for col in ("red", "green", "blue"):
        r = meas[col]
        if not r["ok"]:
            continue
        u, v, z = dc.project(r["center"][None, :])
        draw_cross(dr, (float(u[0]), float(v[0])), CUBE_COL[col])
        g = gt[col][0]
        gu, gv, _ = dc.project(g[None, :])
        # 真实中心用虚线方块标记，和十字重叠即为测量正确
        dr.rectangle([float(gu[0]) - 6, float(gv[0]) - 6,
                      float(gu[0]) + 6, float(gv[0]) + 6],
                     outline=CUBE_COL[col], width=2)
        dr.text((float(u[0]) + 18, float(v[0]) - 10),
                "%s z=%.3f" % (col, r["z_top"]), fill=CUBE_COL[col], font=fs)
    dr.rectangle([0, 0, img.width - 1, 34], fill=(0, 0, 0, 160))
    dr.text((12, 6), title, fill=(255, 255, 255), font=f)
    return img


def scatter_view(meas, gt, pts=None, cols=None, size=480, title="点云俯视 (x-y)"):
    """俯视散点图：真实点云 + 每块方块的测量中心 vs 真实中心。"""
    img = Image.new("RGB", (size, size), (16, 18, 22))
    dr = ImageDraw.Draw(img)
    x0, x1 = -0.45, 0.45
    y0, y1 = -0.68, 0.68

    def to_px(x, y):
        return (int((x - x0) / (x1 - x0) * (size - 1)),
                int((1.0 - (y - y0) / (y1 - y0)) * (size - 1)))

    # 先把真实点云画成淡色底图（每 6 个点取 1 个，够密又不过慢）
    if pts is not None and cols is not None and len(pts):
        sel = slice(None, None, 6)
        p = pts[sel]
        c = cols[sel].astype(int)
        keep = (p[:, 2] > 0.70) & (p[:, 2] < 1.35)     # 桌面以上一小段
        p = p[keep]
        c = c[keep]
        px = ((p[:, 0] - x0) / (x1 - x0) * (size - 1)).astype(int)
        py = ((1.0 - (p[:, 1] - y0) / (y1 - y0)) * (size - 1)).astype(int)
        good = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        buf = np.asarray(img).copy()
        buf[py[good], px[good]] = (c[good] * 0.55 + 40).astype(np.uint8)
        img = Image.fromarray(buf)
        dr = ImageDraw.Draw(img)

    def rect(a, b, **kw):
        """把两个对角点转成 PIL 要求的 [x0,y0,x1,y1]（左上到右下）。"""
        xa, ya = to_px(*a)
        xb, yb = to_px(*b)
        dr.rectangle([min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)], **kw)

    # 桌面轮廓
    rect((-0.40, -0.60), (0.40, 0.60), outline=(95, 100, 112), width=2)
    # 篮子
    rect((-0.30, 0.28), (-0.02, 0.56), outline=(170, 125, 70), width=2)
    for col in ("red", "green", "blue"):
        r = meas[col]
        if not r["ok"]:
            continue
        p = r["center"]
        u, v = to_px(p[0], p[1])
        dr.ellipse([u - 5, v - 5, u + 5, v + 5], fill=CUBE_COL[col])
        g = gt[col][0]
        gu, gv = to_px(g[0], g[1])
        dr.ellipse([gu - 10, gv - 10, gu + 10, gv + 10],
                   outline=CUBE_COL[col], width=2)
        dr.text((u + 10, v - 9), col, fill=CUBE_COL[col], font=font(14))
    dr.rectangle([0, 0, size - 1, 30], fill=(0, 0, 0))
    dr.text((10, 6), title, fill=(225, 225, 225), font=font(16))
    dr.text((10, size - 24), "● 深度相机测量   ○ 真实位姿   灰=桌面/篮子点云",
            fill=(160, 165, 175), font=font(13))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", action="store_true")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    import json
    poses = json.loads((HERE / "poses.json").read_text(encoding="utf-8"))

    render_pose(m, d, poses, "ready")
    dc = DepthCamera(m, d)
    rgb, dep = dc.capture()
    meas = measure(dc, rgb, dep)
    gt = ground_truth(m, d)
    pts, cols3, _ = dc.pointcloud(dep, rgb)

    tiles = [
        annotate(rgb, dc, meas, gt, "① 深度相机 RGB 视图（848x480, fovy 58°）"),
        Image.fromarray(colormap(dep)).resize((848, 480), Image.NEAREST),
        scatter_view(meas, gt, pts, cols3, 480),
    ]
    # 深度伪彩图加个色标说明
    dr = ImageDraw.Draw(tiles[1])
    dr.rectangle([0, 0, 847, 34], fill=(0, 0, 0))
    dr.text((12, 6), "② 深度图（红=近 0.4 m，蓝=远 2.4 m；黑=天空/遮挡）",
            fill=(255, 255, 255), font=font(20))

    big = Image.new("RGB", (848 + 480 + 8, 480 * 2 + 8), (10, 11, 13))
    big.paste(tiles[0], (0, 0))
    big.paste(tiles[1], (0, 488))
    big.paste(tiles[2], (856, 0))
    # 右下角放检测结果表
    panel = Image.new("RGB", (480, 480), (16, 18, 22))
    pdr = ImageDraw.Draw(panel)
    pdr.text((10, 8), "③ 深度测量结果", fill=(220, 220, 220), font=font(18))
    y = 44
    pdr.text((10, y), "颜色   测量中心 (m)            误差", fill=(150, 155, 165), font=font(14))
    y += 26
    for col in ("red", "green", "blue"):
        r = meas[col]
        if not r["ok"]:
            continue
        c = r["center"]
        g = gt[col][0]
        dxy = 1000 * float(np.hypot(c[0] - g[0], c[1] - g[1]))
        pdr.text((10, y), "%-5s [%6.3f %6.3f %6.3f]" % (col, *c), fill=CUBE_COL[col], font=font(14))
        pdr.text((10, y + 18), "      真实 [%6.3f %6.3f %6.3f]  Δxy %.1f mm"
                 % (*g, dxy), fill=(150, 155, 165), font=font(12))
        y += 48
    y += 8
    pdr.text((10, y), "点云 %d 点  桌面 z=%.4f (真值 0.740)" % (len(pts), np.median(
        pts[np.abs(pts[:, 2] - 0.74) < 0.02][:, 2])), fill=(180, 185, 195), font=font(13))
    y += 26
    pdr.text((10, y), "内参 fx=fy=%.1f  cx=%.1f  cy=%.1f" % (dc.fx, dc.width / 2, dc.height / 2),
             fill=(180, 185, 195), font=font(13))
    y += 26
    pdr.text((10, y), "光心 (%.2f, %.2f, %.2f)" % tuple(dc.pose()[0]),
             fill=(180, 185, 195), font=font(13))
    big.paste(panel, (856, 488))

    OUT.mkdir(exist_ok=True)
    big.save(OUT / "depth_camera.png")
    print("写出 out/depth_camera.png（%dx%d）" % big.size)

    if args.multi:
        for name in POSE_NAMES:
            render_pose(m, d, poses, name)
            rgb2, dep2 = dc.capture()
            mm = measure(dc, rgb2, dep2)
            ann = annotate(rgb2, dc, mm, gt, "姿态 %s" % name)
            ann.save(OUT / ("depth_pose_%s.png" % name))
            print("写出 out/depth_pose_%s.png" % name)
    dc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
