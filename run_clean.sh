#!/usr/bin/env bash
# 交付前清理一次性调试脚本（保留有复用价值的诊断工具）
cd /home/zhangb/NERO || exit 1

# 一次性探针：篮子/格位筛选已经定稿，结果写进 README 与代码常量
rm -f tools/probe_slots.py tools/probe_slots2.py tools/probe_basket.py \
      tools/probe_basket2.py tools/diag_all.py

# 重复的 runner（run_e2e.sh / run_rebuild.sh 已覆盖）
rm -f run_all.sh run_cp.sh run_ls.sh

# 根目录一次性裁剪脚本
rm -f crop_px.py

echo "清理后 tools/："
ls tools/
echo
echo "根目录 sh："
ls *.sh
