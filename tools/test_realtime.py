#!/usr/bin/env python3
"""验证 --viewer 路径的实时节拍：带一个 stub viewer（只数 sync 次数）跑完整流程，
比较墙钟时间与仿真时间。期望倍速≈1.0（之前是 82x 快放）。"""
from __future__ import annotations

import pathlib
import sys
import time

import mujoco

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402


class Stub:
    def __init__(self):
        self.n = 0

    def sync(self):
        self.n += 1


def main():
    m, d, qad = pp.load()
    pp.reset(m, d, qad)
    v = Stub()
    t0 = time.perf_counter()
    for item in pp.plan_for("green"):
        kind, keys, grip = item[0], item[1], item[2]
        hold = item[4] if len(item) > 4 else ()
        slow = item[5] if len(item) > 5 else ()
        if kind == "path":
            pp.move_path(m, d, qad, keys, grip, v, hold=hold, slow=slow)
        else:
            pp.dwell(m, d, qad, grip, viewer=v)
    wall = time.perf_counter() - t0
    print("viewer 模式：墙钟 %.2f s   sync 次数 %d   目标 ≈ 仿真时长" % (wall, v.n))
    print("（若墙钟远小于仿真时长 → 仍是快放；若接近 → 实时）")


if __name__ == "__main__":
    main()
