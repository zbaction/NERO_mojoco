#!/usr/bin/env bash
set -u
cd /home/zhangb/NERO || exit 1
rm -f run_final.sh
git config user.name zbaction
git config user.email zbaction@users.noreply.github.com
git add -A
git commit -q -m "chore: 清理一次性 runner，只保留 run.sh / run_e2e.sh / run_rebuild.sh / run_shots_all.sh / run_smoke.sh"
echo "commit 退出码=$?"
echo
git --no-pager log --oneline -3
echo
echo "--- 工作区 ---"
git status --short
echo "(空 = 干净)"
echo
echo "--- 文件清单 ---"
ls *.py *.sh 2>/dev/null
