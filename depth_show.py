"""腕部深度相机 2x2 展示图：RGB / 深度伪彩 / 点云散点 / 数值面板。

    python3 depth_show.py            # green_hover 一张
    python3 depth_show.py --multi    # 三色各一张，另出 3x2 拼版
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import mujoco
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from depth_cam import DepthCamera
from depth_view import (CUBE_OF, HOVER_OF, color_mask, colormap, ground_truth,
                        measure, RGB_CUBE)

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "nero_table_scene.xml"
OUT = HERE / "out"
POSES = HERE / "poses.json"
COL = {"red": (235, 64, 52), "green": (52, 200, 90), "blue": (60, 110, 230)}

# PIL 的 ImageFont.load_default() 是 11px 位图字体，**没有中文字形**——
# 中文字符全被画成空白/方块，面板看着"几乎没内容"。必须显式加载 CJK TrueType。
_CJK = ["/home/zhangb/.local/share/fonts/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
_CJK_BOLD = ["/home/zhangb/.local/share/fonts/NotoSansCJKsc-Bold.otf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]


def _font(size, bold=False):
    for p in (_CJK_BOLD if bold else _CJK):
        if pathlib.Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


F_TITLE = _font(26, bold=True)
F_BODY = _font(20)
F_SMALL = _font(17)


def text(dr, xy, s, fill=(220, 220, 220), font=None):
    dr.text(xy, s, fill=fill, font=font or F_BODY)


def rect(a, b):
    """PIL 的 rectangle 要求 [x0,y0,x1,y1] 且 y1>=y0，这里统一排序。"""
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]


def annotate(rgb, dc, meas, gt, title):
    im = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 32], fill=(0, 0, 0))
    text(dr, (8, 5), title, fill=(255, 235, 120), font=F_BODY)
    # 测量中心（十字）/ 真值（方框）
    for col, r in meas.items():
        if not r.get("ok"):
            continue
        cx, cy, cz = r["center"]
        u, v, _ = dc.project(np.array([[cx, cy, cz]]))
        u, v = float(u[0]), float(v[0])
        if not (0 <= u < im.width and 0 <= v < im.height):
            continue
        dr.line([u - 9, v, u + 9, v], fill=COL[col], width=2)
        dr.line([u, v - 9, u, v + 9], fill=COL[col], width=2)
        g = gt[col][0]
        gu, gv, _ = dc.project(np.array([g]))
        dr.rectangle(rect((gu[0] - 8, gv[0] - 8), (gu[0] + 8, gv[0] + 8)),
                     outline=COL[col], width=1)
        text(dr, (u + 11, v - 6), "%s %.1fmm" % (
            col, 1000 * float(np.hypot(cx - g[0], cy - g[1]))), fill=COL[col], font=F_SMALL)
    return im


def scatter(meas, gt, pts, cols, size=480, title="", window=0.34):
    """俯视散点：以**目标方块为中心**取一个小窗口（默认 0.34 m 见方）。

    腕部相机的点云里绝大部分是桌面平面，一路延伸到地平线；如果按整片点云
    自动取景，方块只剩几个像素。这里固定以测量到的方块中心为窗口中心，
    这样方块顶面的颜色块、桌面平面、测量十字/真值叉号都在同一个尺度上可比。
    """
    im = Image.new("RGB", (size, size), (18, 18, 22))
    px = im.load()
    keep = np.zeros(len(pts), bool) if len(pts) else np.zeros(0, bool)
    sub = pts[keep]
    subc = cols[keep]
    cen = None
    if meas:
        cs = [r["center"] for r in meas.values() if r.get("ok")]
        if cs:
            cen = np.mean(np.array(cs), axis=0)
    if cen is None:
        cen = pts.mean(axis=0) if len(pts) else np.zeros(3)
    half = 0.5 * window
    if len(pts):
        keep = ((np.abs(pts[:, 0] - cen[0]) < half) & (np.abs(pts[:, 1] - cen[1]) < half)
                & (pts[:, 2] > 0.70) & (pts[:, 2] < 0.95))
        sub = pts[keep]
        subc = cols[keep]
    lo = np.array([cen[0] - half, cen[1] - half])
    hi = np.array([cen[0] + half, cen[1] + half])
    if len(sub):
        sx = (sub[:, 0] - lo[0]) / (hi[0] - lo[0]) * (size - 1)
        sy = (sub[:, 1] - lo[1]) / (hi[1] - lo[1]) * (size - 1)
        ok = (sx >= 0) & (sx < size) & (sy >= 0) & (sy < size)
        for x, y, c in zip(sx[ok].astype(int), sy[ok].astype(int), subc[ok]):
            px[x, size - 1 - y] = tuple(int(v * 0.85 + 25) for v in c)
    dr = ImageDraw.Draw(im)

    def T(xy):
        return ((xy[0] - lo[0]) / (hi[0] - lo[0]) * (size - 1),
                size - 1 - (xy[1] - lo[1]) / (hi[1] - lo[1]) * (size - 1))

    # 1 cm 网格，便于目测毫米级偏差
    for d in np.arange(-0.15, 0.151, 0.05):
        a0 = T((cen[0] + d, lo[1]))
        b0 = T((cen[0] + d, hi[1]))
        dr.line([a0, b0], fill=(52, 56, 64))
        a1 = T((lo[0], cen[1] + d))
        b1 = T((hi[0], cen[1] + d))
        dr.line([a1, b1], fill=(52, 56, 64))
    for col, r in meas.items():
        if not r.get("ok"):
            continue
        g = gt[col][0]
        cu, cv = T(r["center"][:2])
        dr.ellipse([cu - 9, cv - 9, cu + 9, cv + 9], outline=COL[col], width=2)
        gu, gv = T(g[:2])
        dr.line([gu - 7, gv, gu + 7, gv], fill=(255, 255, 255), width=2)
        dr.line([gu, gv - 7, gu, gv + 7], fill=(255, 255, 255), width=2)
        dxy = 1000 * float(np.hypot(r["center"][0] - g[0], r["center"][1] - g[1]))
        text(dr, (8, 30), "%s: ● 测得   ✛ 真值   Δ=%.1f mm" % (col, dxy),
             fill=COL[col], font=F_SMALL)
    if title:
        dr.rectangle([0, 0, size, 26], fill=(0, 0, 0))
        text(dr, (8, 4), title, fill=(230, 230, 230), font=F_SMALL)
        text(dr, (8, size - 24), "窗口 %.2f × %.2f m，格距 50 mm，点 %d"
             % (hi[0] - lo[0], hi[1] - lo[1], keep.sum() if len(pts) else 0),
             fill=(180, 180, 180), font=F_SMALL)
    return im


def panel(meas, gt, ok_all, pose, cam_body):
    im = Image.new("RGB", (848, 480), (32, 34, 40))
    dr = ImageDraw.Draw(im)
    y = 26
    text(dr, (24, y), "腕部深度相机（eye-in-hand）测量结果", fill=(255, 230, 90), font=F_TITLE)
    y += 44
    text(dr, (24, y), "姿态: %s" % pose, fill=(210, 210, 210))
    y += 28
    text(dr, (24, y), "相机挂载: %s（随手运动）" % cam_body, fill=(210, 210, 210))
    y += 40
    text(dr, (24, y), "%-6s %9s %11s %11s" % ("颜色", "像素", "Δxy", "Δz"),
         fill=(140, 190, 255))
    y += 30
    for col in ("red", "green", "blue"):
        r = meas.get(col, {})
        if not r.get("ok"):
            text(dr, (24, y), "%-6s %9s" % (col, "不可见"), fill=(130, 130, 130))
            y += 30
            continue
        c = r["center"]
        g = gt[col][0]
        dxy = float(np.hypot(c[0] - g[0], c[1] - g[1]))
        dz = float(c[2] - (g[2] + gt[col][1]))
        text(dr, (24, y), "%-6s %9d %8.1fmm %8.1fmm" % (col, r["px"], 1000 * dxy, 1000 * dz),
             fill=COL[col])
        y += 30
    y += 20
    text(dr, (24, y), "结论: %s" % ("PASS" if ok_all else "FAIL"),
         fill=(80, 220, 120) if ok_all else (240, 90, 90), font=F_TITLE)
    y += 52
    for line in ("· 光心固定在 gripper_base 侧面 95 mm 支架上，沿轴退后 20 mm",
                 "· 光轴朝工具轴侧倾 46°，fovy 80°，848×480，量程 0.06~1.0 m",
                 "· 每个姿态只看到目标方块 + 局部桌面（腕部相机的固有特性）",
                 "· 彩色圆环 = 深度测量中心，白色叉号 = 真值中心"):
        text(dr, (24, y), line, fill=(168, 168, 168), font=F_SMALL)
        y += 26
    return im


def shot(m, d, dc, adr, poses, col):
    mujoco.mj_resetData(m, d)
    d.qpos[adr] = np.radians(np.array(poses[HOVER_OF[col]]))
    d.qpos[7] = d.qpos[8] = 0.05
    mujoco.mj_forward(m, d)
    rgb, dep = dc.capture()
    meas = measure(dc, rgb, dep, cols=(col,))
    gt = ground_truth(m, d)
    r = meas[col]
    ok = bool(r.get("ok"))
    if ok:
        c = r["center"]
        dxy = float(np.hypot(c[0] - gt[col][0][0], c[1] - gt[col][0][1]))
        dz = float(c[2] - (gt[col][0][2] + gt[col][1]))
        ok = dxy < 0.008 and abs(dz) < 0.012
    pts, pcols, _ = dc.pointcloud(dep, rgb)
    a = annotate(rgb, dc, meas, gt, "%s  %s_hover  %d px" % (col, col, r.get("px", 0)))
    b = Image.fromarray(colormap(dep, 0.10, 0.90)).convert("RGB")
    bb = ImageDraw.Draw(b)
    bb.rectangle([0, 0, 200, 22], fill=(0, 0, 0))
    bb.text((6, 5), "depth 0.10~0.90 m", fill=(255, 255, 0))
    c2 = scatter(meas, gt, pts, pcols, 480, "top view (world xy, 腕部点云)")
    e = panel(meas, gt, ok, HOVER_OF[col], "gripper_base")
    sheet = Image.new("RGB", (1696, 960), (0, 0, 0))
    sheet.paste(a, (0, 0))
    sheet.paste(b, (848, 0))
    # scatter 是正方形，左下象限是 848x480，居中放并把两侧留成深色，避免被裁掉一半
    quad = Image.new("RGB", (848, 480), (12, 12, 14))
    quad.paste(c2, ((848 - 480) // 2, 0))
    sheet.paste(quad, (0, 480))
    sheet.paste(e, (848, 480))
    return sheet, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", action="store_true")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    poses = json.loads(POSES.read_text(encoding="utf-8"))
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)]
           for i in range(1, 8)]
    dc = DepthCamera(m, d, cam_name="wrist_cam", width=848, height=480, max_depth=1.0)

    OUT.mkdir(exist_ok=True)
    sheets, allok = [], True
    for col in (("red", "green", "blue") if args.multi else ("green",)):
        s, ok = shot(m, d, dc, adr, poses, col)
        allok = allok and ok
        sheets.append((col, s))
        print("%-6s %s -> out/wrist_depth_%s.png" % (col, "PASS" if ok else "FAIL", col))
    for col, s in sheets:
        s.save(OUT / ("wrist_depth_%s.png" % col))
    if len(sheets) > 1:
        big = Image.new("RGB", (848, 480 * len(sheets)), (0, 0, 0))
        for i, (_, s) in enumerate(sheets):
            big.paste(s, (0, 480 * i))
        big.save(OUT / "wrist_depth_all.png")
        print("拼版 -> out/wrist_depth_all.png")
    dc.close()
    print("结果: %s" % ("PASS" if allok else "FAIL"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
