#!/usr/bin/env bash
# 快速烟测：交互窗口路径能不能正常起来（3 秒后自己关）
set -u
PY=/home/zhangb/so101_mujoco/.venv/bin/python
cd /home/zhangb/NERO || exit 1
$PY -u tools/smoke_viewer.py
echo "smoke 退出码=$?"
