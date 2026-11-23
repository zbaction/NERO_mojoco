#!/usr/bin/env bash
# 一键入口
#   ./run.sh             交互窗口（机械臂 + 桌面 + 三色方块）
#   ./run.sh demo        交互窗口，自动演示一遍抓取
#   ./run.sh check       无窗口自检（抓取成功返回 0）
#   ./run.sh shots       重出成果图到 out/
#   ./run.sh detail      重出夹爪特写图到 out/
#   ./run.sh validate    跑 6 段全量校验
#   ./run.sh desk        只看第一版的桌子+三色方块
set -e
cd "$(dirname "$0")"
PY=/home/zhangb/so101_mujoco/.venv/bin/python

case "${1:-}" in
  demo)     shift; exec "$PY" view_scene.py --demo "$@" ;;
  check)    shift; exec "$PY" view_scene.py --check "$@" ;;
  shots)    shift; exec "$PY" render_showcase.py "$@" ;;
  detail)   shift; exec "$PY" render_detail.py "$@" ;;
  validate) shift; exec "$PY" validate_nero.py "$@" ;;
  desk)     shift; exec "$PY" view.py "$@" ;;
  *)        exec "$PY" view_scene.py "$@" ;;
esac
