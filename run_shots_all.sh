#!/usr/bin/env bash
set -u
PY=/home/zhangb/so101_mujoco/.venv/bin/python
cd /home/zhangb/NERO || exit 1
echo "=== 三色过程图（顺序 red -> green -> blue，共 21 张） ==="
$PY -u pick_place.py --all --shots
echo "退出码=$?"
