#!/usr/bin/env python3
"""把出图裁到篮子区域放大，便于目视确认。"""
from __future__ import annotations

import pathlib
import sys

from PIL import Image

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
x0, y0, x1, y1 = (int(v) for v in sys.argv[3:7])
im = Image.open(src).crop((x0, y0, x1, y1))
im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
im.save(dst)
print("写出 %s  %dx%d" % (dst, im.width, im.height))
