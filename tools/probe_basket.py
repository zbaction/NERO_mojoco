#!/usr/bin/env python3
"""为篮子挑位置：对若干候选中心解放置位姿，看可达性与关节余量。"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import solve_poses as sp  # noqa: E402

TABLE_Z = 0.74
GRIP_CENTER_Z = TABLE_Z + 0.010 + 0.040      # 篮子底板顶面 + 半个方块
TRANSIT_Z = TABLE_Z + 0.300

# 绿方块的位置（当前抓取目标）
cg, yg = sp.cube_pose("cube_green")
print("绿方块中心 %s  偏航 %.1f deg\n" % (np.round(cg, 3), math.degrees(yg)))

CANDIDATES = [(-0.10, 0.40), (-0.05, 0.44), (0.00, 0.45),
              (0.06, 0.44), (0.10, 0.42), (0.16, 0.45),
              (0.00, -0.44), (0.10, -0.44)]

print("%-16s %-10s %-8s %-8s %-8s %s"
      % ("篮子中心(x,y)", "距离基座", "place", "hover", "transit", "place 关节余量"))
for bx, by in CANDIDATES:
    dist = math.hypot(bx - (-0.30), by - 0.0)
    place_pt = np.array([bx, by, GRIP_CENTER_Z])
    hover_pt = np.array([bx, by, GRIP_CENTER_Z + 0.20])
    mid = np.array([(cg[0] + bx) / 2, (cg[1] + by) / 2, TRANSIT_Z])
    rp = sp.pick(place_pt, seed=7, yaw_tgt=yg)
    rh = sp.pick(hover_pt, seed=8, yaw_tgt=yg)
    rt = sp.pick(mid, seed=9, yaw_tgt=yg)
    if rp is None:
        print("%-16s %-10.3f  place 无解" % ("(%.2f,%.2f)" % (bx, by), dist))
        continue
    q = rp[1]
    margin = math.degrees(np.minimum(q - sp.LO, sp.HI - q).min())
    print("%-16s %-10.3f %-8s %-8s %-8s %.1f deg"
          % ("(%.2f,%.2f)" % (bx, by), dist,
             "ok" if rp else "无解", "ok" if rh else "无解", "ok" if rt else "无解",
             margin))
    if margin < 8:
        print("     <<< 余量偏小，换位置")
