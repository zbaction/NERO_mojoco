#!/usr/bin/env python3
"""检查 19 个关键帧之间的关节空间连续性（相邻位姿最大关节差）。"""
from __future__ import annotations

import json
import pathlib

import numpy as np

P = pathlib.Path.home() / "NERO" / "poses.json"
p = json.loads(P.read_text(encoding="utf-8"))

READY = np.array(p["ready"])
print("ready: " + " ".join("%+.1f" % v for v in READY))
print()
PH = ("hover", "grasp", "lift", "transit", "place_hover", "place")
for col in ("red", "green", "blue"):
    prev = READY
    name = "ready"
    worst = 0.0
    print("%s:" % col)
    for ph in PH:
        q = np.array(p["%s_%s" % (col, ph)])
        d = np.abs(q - prev).max()
        worst = max(worst, d)
        flag = "  <-- 过大" if d > 60 else ""
        print("  %-12s<-%-12s 最大关节跳变 %6.1f°%s"
              % (ph, name, d, flag))
        prev, name = q, ph
    d = np.abs(prev - READY).max()
    print("  %-12s<-%-12s 最大关节跳变 %6.1f°" % ("ready", name, d))
    print("  链条内最大跳变 %.1f°%s" % (max(worst, d),
                                    "  ** 需要改路径 **" if max(worst, d) > 90 else ""))
    print()

print("joint5 一览（检查是否有正负分支切换）：")
for k in ("ready",):
    print("  %-18s %+7.1f" % (k, p[k][4]))
for col in ("red", "green", "blue"):
    for ph in PH:
        k = "%s_%s" % (col, ph)
        print("  %-18s %+7.1f" % (k, p[k][4]))
