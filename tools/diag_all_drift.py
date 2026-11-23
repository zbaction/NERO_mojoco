#!/usr/bin/env python3
"""量化 `--all` 过程中，**已放置**方块是否被后续搬运扰动。

只追踪"已经搬完的"方块：搬第 i 块时，前 i-1 块必须原地不动。
"""
from __future__ import annotations

import math
import pathlib
import sys

import mujoco

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402


def pos(m, d, col):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pp.CUBE_OF[col])
    return d.xpos[b].copy()


def main():
    m, d, qad = pp.load()
    mujoco.mj_forward(m, d)
    print("桌面上三块方块初始位置：")
    for c in pp.COLORS:
        print("  %-5s [%.3f %.3f %.3f]" % (c, *pos(m, d, c)))
    print()

    placed = []          # 已搬完的颜色
    before = {}          # 每次搬之前，已放置方块的位置
    worst = 0.0
    allok = True

    for col in pp.COLORS:
        if placed:
            before = {c: pos(m, d, c) for c in placed}
        ok = pp.run(m, d, qad, col, verbose=False)
        allok = allok and ok
        print("搬完 %-6s -> %s" % (pp.LABEL[col], "PASS" if ok else "FAIL"))

        if before:
            print("    此前已放好的方块有没有被动到：")
            for c, p0 in before.items():
                p1 = pos(m, d, c)
                dn = 1000 * float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                dz = 1000 * float(p1[2] - p0[2])
                flag = "OK" if dn < 5.0 else "**动了**"
                print("      %-5s 水平位移 %.2f mm  dz %+.2f mm  %s"
                      % (c, dn, dz, flag))
                worst = max(worst, dn)
        placed.append(col)
        print()

    print("=" * 62)
    print("已放置方块的最大水平扰动：%.2f mm（限 5.0）" % worst)
    print()
    print("最终三块 vs 各自格位：")
    for c in pp.COLORS:
        ok, p, sc, ft, inner = pp.cube_in_basket(m, d, pp.CUBE_OF[c])
        dx = 1000 * math.hypot(p[0] - sc[0], p[1] - sc[1])
        print("  %-5s [%.3f %.3f %.3f]  格位 (%.3f, %.3f)  偏 %.1f mm  %s"
              % (c, *p, sc[0], sc[1], dx, "PASS" if ok else "FAIL"))
        allok = allok and ok
    print("总结果：%s" % ("PASS" if allok and worst < 5.0 else "FAIL"))
    return 0 if (allok and worst < 5.0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
