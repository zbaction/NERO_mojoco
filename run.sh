#!/usr/bin/env bash
# 一键运行：默认打开交互窗口，带 --render 则只离屏渲染 png
set -e
cd "$(dirname "$0")"
PY=/home/zhangb/so101_mujoco/.venv/bin/python
exec "$PY" view.py "$@"