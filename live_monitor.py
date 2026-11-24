#!/usr/bin/env python3
"""NERO 单窗口实时监控：3D 场景 + 腕部深度相机（eye-in-hand）同框。

一个窗口里同时显示：

    ┌──────────────────────────────┬───────────────────┐
    │                              │  腕部相机 RGB      │
    │   3D 场景（可拖拽旋转/缩放）    ├───────────────────┤
    │                              │  腕部深度（伪彩）    │
    └──────────────────────────────┴───────────────────┘
    [状态栏：时间 / 帧率 / 7 个关节角 / 三块方块位置]
    [抓红] [抓绿] [抓蓝] [全部搬完] [回待命] [暂停] [停止]

用法：
    python3 live_monitor.py                 # 打开监控窗口
    python3 live_monitor.py --auto green    # 打开窗口后自动搬绿方块
    python3 live_monitor.py --auto all      # 依次搬三块
    python3 live_monitor.py --smoke 6       # 跑 6 s 自动退出（自检用）
    python3 live_monitor.py --save out/x.png --smoke 3   # 顺手存一张界面合成图

为什么不直接用 mujoco.viewer：
    `mujoco.viewer.launch_passive` 只能显示 3D 主视图，没有任何 API 能把
    第二台相机的画面嵌进同一个窗口。所以这里自己用 `mujoco.Renderer`
    渲染两个机位，再把三块图贴进同一个 Tk 窗口。

性能（WSL2 + WSLg，D3D12 / Intel Iris Xe，实测 tools/bench_*.py）：
    * `<global shadowsize="4096">` 下 848x480 单帧 **225 ms**（4.4 fps）；
      把 `m.vis.quality.shadowsize` 设为 0 后只要 **43 ms**（23 fps）。
      阴影 pass 要额外重绘一遍 21 万个三角面，是唯一的瓶颈，且与分辨率
      几乎无关（320x180 与 848x480 都是 ~150 ms）。
      → 实时窗口里**关掉阴影**；静态成果图（render_showcase.py 等）仍保留阴影。
    * 3D 1000x620 + 腕部 RGB 500x306 + 深度 500x306 ≈ 65 ms/帧 → 约 12-15 fps。

线程模型（重要）：
    所有按钮回调只往 `self._pending` 里塞一个请求，真正的任务在
    `loop()` 里启动。**不能**在按钮回调里直接跑任务 —— 那个回调是在
    `root.update()` 内部触发的，而任务执行期间又会调 `viewer.sync()`
    → `root.update()`，形成重入。
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time

import numpy as np
import mujoco

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageTk as ImageTk

import tkinter as tk

import pick_place as pp
from depth_view import colormap as depth_colormap

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "nero_table_scene.xml"
OUT = HERE / "out"

FONT_DIR = pathlib.Path.home() / ".local" / "share" / "fonts"
BG = "#0d0f12"
FG_HINT = "#8fa4bb"
FG_TEXT = "#cfe0f0"


def _font(size, bold=False):
    """Tk 默认位图字体没有中文字形，必须显式指定 Noto Sans CJK。"""
    name = "NotoSansCJKsc-Bold.otf" if bold else "NotoSansCJKsc-Regular.otf"
    try:
        ImageFont.truetype(str(FONT_DIR / name), size)
        return ("Noto Sans CJK SC", size)
    except Exception:
        return ("TkDefaultFont", size)


def arm_bounds(m, d):
    """机械臂（geomgroup 2 的 mesh）世界包围盒，用于初始取景。"""
    pts = []
    for g in range(m.ngeom):
        if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        R = d.geom_xmat[g].reshape(3, 3)
        pts.append(m.mesh_vert[a:a + n] @ R.T + d.geom_xpos[g])
    if not pts:
        return np.zeros(3), np.ones(3)
    P = np.vstack(pts)
    return P.min(0), P.max(0)


def depth_to_rgb(dep, lo=0.10, hi=0.90, max_depth=1.0):
    """深度 -> 伪彩。背景（天空盒 = extent*zfar，本场景 63）涂深灰。"""
    cm = depth_colormap(dep, lo=lo, hi=hi)
    bad = (~np.isfinite(dep)) | (dep <= 0.02) | (dep > max_depth)
    cm[bad] = (18, 20, 24)
    return cm


def default_camera(m, d, pad=1.25, az=58.0, el=-20.0):
    """按机械臂实际包围盒构造一个 free 相机（交互窗口默认视线在世界原点，
    会把台面上的机械臂顶出画面，见 README 的排错章节）。"""
    lo, hi = arm_bounds(m, d)
    c = (lo + hi) / 2.0
    span = float(np.linalg.norm(hi - lo))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = c
    cam.distance = pad * span / (
        2.0 * math.tan(math.radians(m.vis.global_.fovy) / 2))
    cam.azimuth = az
    cam.elevation = el
    return cam, c, span


class Abort(Exception):
    """任务执行过程中点了"停止"。"""


class Monitor:
    """既是 Tk 窗口，也是 pick_place 眼里的 "viewer"（提供 sync / is_running）。"""

    def __init__(self, m, d, qad, w3d=1000, h3d=620, wc=500, hc=306, fps=12.0):
        self.m, self.d, self.qad = m, d, qad
        self.fps = float(fps)
        self.alive = True
        self.paused = False
        self.busy = False
        self._abort = False
        self._pending = None          # 待执行的抓取请求："red"/"green"/"blue"/"all"
        self._last_draw = 0.0
        self._draw_dt = []
        self.msg = ""
        self.w3d, self.h3d, self.wc, self.hc = w3d, h3d, wc, hc
        self._last_c = None
        self._frame_no = 0

        # 阴影是唯一瓶颈（见模块 docstring）：实时窗口里关掉
        m.vis.quality.shadowsize = 0
        m.vis.quality.offsamples = 0
        m.light_castshadow[:] = 0
        # 实时窗口里把 headlight 提亮：静态成果图用的是 3 盏方向光 + 弱头灯，
        # 在 1000x620 的小画面上显得偏暗（实测面板亮度中位 70/255）。
        m.vis.headlight.ambient[:] = (0.42, 0.43, 0.46)
        m.vis.headlight.diffuse[:] = (0.52, 0.52, 0.53)
        m.vis.headlight.specular[:] = (0.10, 0.10, 0.10)

        self.r3 = mujoco.Renderer(m, h3d, w3d)
        self.rc = mujoco.Renderer(m, hc, wc)
        self.wcid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        if self.wcid < 0:
            raise RuntimeError("模型里没有 wrist_cam 相机")

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.reset_camera()

        self._build_ui()

    # ---------------------------------------------------------------- 相机
    def reset_camera(self):
        cam, c, span = default_camera(self.m, self.d)
        self.cam.lookat[:] = cam.lookat
        self.cam.distance = cam.distance
        self.cam.azimuth = cam.azimuth
        self.cam.elevation = cam.elevation
        self._last_draw = 0.0
        return c, span

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        w3d, h3d, wc, hc = self.w3d, self.h3d, self.wc, self.hc
        self.root = tk.Tk()
        self.root.title("NERO 实时监控 · 3D 场景 + 腕部深度相机")
        self.root.configure(bg=BG)

        top = tk.Frame(self.root, bg=BG)
        top.pack(padx=8, pady=(8, 4))

        left = tk.Frame(top, bg=BG)
        left.grid(row=0, column=0, sticky="n")
        tk.Label(left, text="3D 场景　左键拖拽旋转 / 滚轮缩放 / 右键单击复位",
                 bg=BG, fg=FG_HINT, font=_font(10)).pack(anchor="w")
        self.lbl3d = tk.Label(left, bg="#101216", bd=0)
        self.lbl3d.pack()
        self.lbl3d.bind("<ButtonPress-1>", self._drag_start)
        self.lbl3d.bind("<B1-Motion>", self._drag_move)
        self.lbl3d.bind("<ButtonPress-3>", lambda e: self.reset_camera())
        self.lbl3d.bind("<MouseWheel>", self._wheel)
        self.lbl3d.bind("<Button-4>", lambda e: self._zoom(1.1))
        self.lbl3d.bind("<Button-5>", lambda e: self._zoom(1.0 / 1.1))

        right = tk.Frame(top, bg=BG)
        right.grid(row=0, column=1, sticky="n", padx=(10, 0))
        tk.Label(right, text="腕部相机 RGB（eye-in-hand）", bg=BG, fg=FG_HINT,
                 font=_font(10)).pack(anchor="w")
        self.lblrgb = tk.Label(right, bg="#101216", bd=0)
        self.lblrgb.pack()
        tk.Label(right, text="腕部深度（红=近 / 蓝=远）", bg=BG, fg=FG_HINT,
                 font=_font(10)).pack(anchor="w", pady=(6, 0))
        self.lbldep = tk.Label(right, bg="#101216", bd=0)
        self.lbldep.pack()

        self.status = tk.StringVar(value="启动中…")
        tk.Label(self.root, textvariable=self.status, bg=BG, fg=FG_TEXT,
                 font=_font(10), anchor="w", justify="left").pack(
                     fill="x", padx=10, pady=(2, 2))

        bar = tk.Frame(self.root, bg=BG)
        bar.pack(padx=8, pady=(2, 8), fill="x")
        self.btns = {}
        for text, cmd in (("抓红", lambda: self.request("red")),
                          ("抓绿", lambda: self.request("green")),
                          ("抓蓝", lambda: self.request("blue")),
                          ("全部搬完", lambda: self.request("all")),
                          ("回待命", self.go_ready),
                          ("暂停", self.toggle_pause),
                          ("停止", self.stop)):
            b = tk.Button(bar, text=text, command=cmd, width=9, bg="#1b2430",
                          fg="#dce8f4", activebackground="#2a3a4d",
                          activeforeground="#ffffff", relief="flat",
                          font=_font(10))
            b.pack(side="left", padx=3)
            self.btns[text] = b

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._photos = (None, None, None)
        self.root.update()
        print("窗口已构建（贴图前 %d x %d，贴图后会自动撑到内容尺寸）"
              % (self.root.winfo_width(), self.root.winfo_height()))

    def _drag_start(self, ev):
        self._drag = (ev.x, ev.y)

    def _drag_move(self, ev):
        if not getattr(self, "_drag", None):
            return
        dx, dy = ev.x - self._drag[0], ev.y - self._drag[1]
        self._drag = (ev.x, ev.y)
        self.cam.azimuth = (self.cam.azimuth - 0.5 * dx) % 360.0
        self.cam.elevation = float(np.clip(self.cam.elevation + 0.4 * dy, -89.0, 89.0))
        self._last_draw = 0.0

    def _wheel(self, ev):
        self._zoom(1.1 if ev.delta > 0 else 1.0 / 1.1)

    def _zoom(self, k):
        self.cam.distance = float(np.clip(self.cam.distance / k, 0.30, 6.00))
        self._last_draw = 0.0

    # ---------------------------------------------------------------- 绘制
    def _draw(self, force_depth=False):
        t0 = time.perf_counter()
        self.r3.update_scene(self.d, self.cam)
        a = self.r3.render()

        self.rc.update_scene(self.d, camera=self.wcid)
        b = self.rc.render()

        # 深度图比 RGB 便宜但仍有 ~13 ms 固定开销；交互时每 2 帧更新一次，
        # 3D 与腕部 RGB 保持每帧更新（bench_onectx.py：深度≈13 ms，3D≈42，RGB≈44）。
        self._frame_no = getattr(self, "_frame_no", 0) + 1
        if force_depth or (self._frame_no % 2 == 0) or self._last_c is None:
            self.rc.enable_depth_rendering()
            self.rc.update_scene(self.d, camera=self.wcid)
            z = self.rc.render()
            self.rc.disable_depth_rendering()
            c = depth_to_rgb(z)
            self._last_c = c
        else:
            c = self._last_c

        # 复用 PhotoImage：只创建一次，之后 paste 覆盖，省去每帧 3 次构造开销
        if self._photos[0] is None:
            self._photos = (ImageTk.PhotoImage(Image.fromarray(a)),
                            ImageTk.PhotoImage(Image.fromarray(b)),
                            ImageTk.PhotoImage(Image.fromarray(c)))
            self.lbl3d.configure(image=self._photos[0])
            self.lblrgb.configure(image=self._photos[1])
            self.lbldep.configure(image=self._photos[2])
        else:
            self._photos[0].paste(Image.fromarray(a))
            self._photos[1].paste(Image.fromarray(b))
            self._photos[2].paste(Image.fromarray(c))

        self._draw_dt = (self._draw_dt + [time.perf_counter() - t0])[-30:]
        self._last_draw = time.perf_counter()
        return a, b, c

    def _maybe_draw(self):
        if time.perf_counter() - self._last_draw >= 1.0 / self.fps:
            self._draw()

    def _status(self):
        dts = self._draw_dt
        fps = 1.0 / (sum(dts) / len(dts)) if dts else 0.0
        q = np.degrees([self.d.qpos[a] for a in self.qad])
        pos = []
        for col in pp.COLORS:
            bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, pp.CUBE_OF[col])
            p = self.d.xpos[bid]
            pos.append("%s[%.3f %.3f %.3f]" % (pp.LABEL[col][0], p[0], p[1], p[2]))
        grip = float(self.d.ctrl[7]) if self.m.nu > 7 else 0.0
        state = "任务中" if self.busy else ("暂停" if self.paused else "空闲")
        head = "t=%6.2f s | %s | 绘制 %4.1f fps | 夹爪开度 %5.1f mm" % (
            self.d.time, state, fps, 2000.0 * grip)
        if self.msg:
            head += " | " + self.msg
        return "%s\n关节(deg) %s\n%s" % (
            head, " ".join("%6.1f" % v for v in q), "  ".join(pos))

    # ---------------------------------------------------------------- viewer 接口
    def sync(self):
        """pick_place 的 _step_loop 每 ~1/60 s 调一次。"""
        if not self.alive:
            raise Abort()
        self.root.update()
        if self._abort:
            self._abort = False
            raise Abort()
        self._maybe_draw()
        self.status.set(self._status())

    def is_running(self):
        return self.alive

    # ---------------------------------------------------------------- 动作派发
    def request(self, what):
        """按钮回调只记下请求，真正的执行放在 loop() 里（避免 Tk 重入）。"""
        if self.busy:
            self.msg = "正在执行任务，稍候…"
            return
        self._pending = what

    def _set_enabled(self, on):
        for b in self.btns.values():
            b.configure(state="normal" if on else "disabled")

    def _run(self, what):
        self.busy = True
        self._set_enabled(False)
        cols = list(pp.COLORS) if what == "all" else [what]
        try:
            for col in cols:
                print("\n=== 抓取 %s ===" % pp.LABEL[col])
                ok = pp.run(self.m, self.d, self.qad, col, viewer=self)
                self.msg = "%s %s" % (pp.LABEL[col], "已放进篮子" if ok else "任务失败")
        except Abort:
            self.msg = "已中止"
            print("\n已中止")
        finally:
            self.busy = False
            self._set_enabled(True)

    def go_ready(self):
        if self.busy:
            return
        pp.reset(self.m, self.d, self.qad)
        mujoco.mj_forward(self.m, self.d)
        self.msg = "已回到待命姿态"
        self._draw()

    def toggle_pause(self):
        self.paused = not self.paused
        self.btns["暂停"].configure(text="继续" if self.paused else "暂停")

    def stop(self):
        if self.busy:
            self._abort = True

    def close(self):
        self.alive = False
        try:
            self.root.destroy()
        except Exception:
            pass

    # ---------------------------------------------------------------- 主循环
    def loop(self, seconds=None):
        t_start = time.perf_counter()
        last = t_start
        acc = 0.0
        self._draw()
        # 贴完图后按内容请求尺寸把窗口定死，避免拖拽时窗口跳动
        self.root.update_idletasks()
        self.root.geometry("%dx%d" % (self.root.winfo_reqwidth(),
                                      self.root.winfo_reqheight()))
        self.root.update()
        print("窗口客户区 %d x %d 像素"
              % (self.root.winfo_width(), self.root.winfo_height()))
        while self.alive:
            now = time.perf_counter()
            dt = min(now - last, 0.12)
            last = now
            if self._pending and not self.busy:
                what, self._pending = self._pending, None
                self._run(what)
                continue
            if not self.paused and not self.busy:
                acc += dt
                n = int(acc / self.m.opt.timestep)
                if n > 0:
                    mujoco.mj_step(self.m, self.d, n)
                    acc -= n * self.m.opt.timestep
            self.root.update()
            self._maybe_draw()
            self.status.set(self._status())
            if seconds is not None and now - t_start >= seconds:
                break
            time.sleep(0.002)


def load(scene=SCENE):
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    qad = pp.qad_of(m)
    k = m.key("ready").qpos
    d.qpos[qad] = k[qad]
    d.qpos[7] = d.qpos[8] = pp.GRIP_OPEN
    d.ctrl[:7] = k[qad]
    d.ctrl[7] = d.ctrl[8] = pp.GRIP_OPEN
    mujoco.mj_forward(m, d)
    return m, d, qad


def compose(frames, status="", path=""):
    """把三块画面 + 状态栏文本拼成一张图（不经过窗口，尺寸精确）。"""
    a, b, c = frames
    h = a.shape[0]
    right = np.vstack([b, c])
    if right.shape[0] < h:
        pad = np.full((h - right.shape[0], right.shape[1], 3), 18, np.uint8)
        right = np.vstack([right, pad])
    right = right[:h]
    gap = np.full((h, 8, 3), 13, np.uint8)
    body = np.hstack([a, gap, right])

    bar_h = 74
    img = Image.new("RGB", (body.shape[1], h + bar_h), (13, 15, 18))
    img.paste(Image.fromarray(body), (0, 0))
    dr = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(str(FONT_DIR / "NotoSansCJKsc-Bold.otf"), 20)
    f_body = ImageFont.truetype(str(FONT_DIR / "NotoSansCJKsc-Regular.otf"), 16)
    dr.text((10, h + 6), "NERO 实时监控", font=f_title, fill=(210, 228, 245))
    for i, line in enumerate(status.split("\n")[:-1]):
        dr.text((10, h + 30 + 18 * i), line, font=f_body, fill=(150, 172, 194))
    if path:
        OUT.mkdir(exist_ok=True)
        img.save(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="NERO 单窗口实时监控")
    ap.add_argument("--auto", choices=list(pp.COLORS) + ["all"], default=None,
                    help="窗口打开后自动执行抓取任务")
    ap.add_argument("--smoke", type=float, default=None, help="跑这么多秒自动退出（自检）")
    ap.add_argument("--save", default="", help="退出前把界面合成图存成 png")
    ap.add_argument("--fps", type=float, default=12.0, help="绘制帧率上限（默认 12）")
    a = ap.parse_args()

    m, d, qad = load()
    mon = Monitor(m, d, qad, fps=a.fps)
    print("窗口已打开：%d body / %d geom / %d 关节，wrist_cam fovy=%.0f°"
          % (m.nbody, m.ngeom, m.njnt, m.cam_fovy[mon.wcid]))

    if a.auto:
        mon.request(a.auto)

    mon.loop(seconds=a.smoke)
    if mon._draw_dt:
        avg = sum(mon._draw_dt) / len(mon._draw_dt)
        print("绘制帧率 %.1f fps（单帧 %.1f ms，最近 %d 次平均）"
              % (1.0 / avg, avg * 1e3, len(mon._draw_dt)))
    if a.save:
        fr = mon._draw(force_depth=True)
        print("界面合成图 -> %s" % compose(fr, mon._status(), a.save))
    mon.close()
    return 0


if __name__ == "__main__":
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
        os._exit(rc if isinstance(rc, int) else 0)   # 规避 WSLg 退出时 GLFW segfault
