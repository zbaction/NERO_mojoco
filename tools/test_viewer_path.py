#!/usr/bin/env python3
"""模拟 `./run.sh pick all`（--viewer 路径），用假 viewer 替代真窗口。

--viewer 相对 --check 只多一个 v.sync() 调用，逻辑完全一样；
这里用 stub 覆盖 sync()，确认三色连续搬运在前台路径也成立。
"""
from __future__ import annotations

import pathlib
import sys

import mujoco

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402


class StubViewer:
    def __init__(self):
        self.n = 0

    def sync(self):
        self.n += 1


def main():
    m, d, qad = pp.load()
    v = StubViewer()
    mujoco.mj_forward(m, d)
    allok = True
    for col in pp.COLORS:
        ok = pp.run(m, d, qad, col, viewer=v, verbose=False)
        sc = pp.cube_in_basket(m, d, pp.CUBE_OF[col])
        p = sc[1]
        print("  %-5s 落点 [%.3f %.3f %.3f]  -> %s"
              % (col, *p, "PASS" if ok else "FAIL"))
        allok = allok and ok
    print()
    print("viewer.sync() 被调用 %d 次（说明走的是 --viewer 分支）" % v.n)
    print("窗口路径三色连载：%s" % ("PASS" if allok else "FAIL"))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
