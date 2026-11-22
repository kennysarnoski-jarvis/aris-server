#!/bin/zsh
# The DSpark speed run on the M4, from ~/inkling-work (a copy of this worktree's src/ and bench/). The draft head is
# mtp.safetensors in the model directory. Refuses while another GPU job runs; stops Aris Server only while idle; the
# EXIT trap always restores it.
#   zsh bench/m4_dspark_speed.sh --model ~/models/inkling-2-mini-mlx-4bit --prompts bench/prompts.json \
#       --out runs/dspark-speed-greedy.json
set -u
cd ~/inkling-work || exit 9
U=$(id -u); PLIST=~/Library/LaunchAgents/dev.aris.server.plist
mkdir -p runs
LOG=runs/dspark-speed.log
stamp() { date '+%F %T'; }
busy() { [ "$(lsof -nP -iTCP:8484 -sTCP:ESTABLISHED 2>/dev/null | grep -c ESTABLISHED)" -gt 0 ]; }
others() { pgrep -fl 'oMLX|omlx|opencode|run_native_trial|hard_pilot|agent_loop|speed_mtp|speed_dspark|armbp_server|hold_gpu' | grep -v -e pgrep -e "$$"; }
if [ -n "$(others)" ]; then echo "$(stamp) another GPU job is running, not starting: $(others | head -3 | tr '\n' ';')" | tee -a $LOG; exit 3; fi
if busy; then echo "$(stamp) Aris Server is busy, not stopping it" | tee -a $LOG; exit 2; fi
restore() {
  launchctl bootstrap gui/$U $PLIST 2>/dev/null; launchctl kickstart gui/$U/dev.aris.server 2>/dev/null
  echo "$(stamp) Aris Server restored" | tee -a $LOG
}
trap restore EXIT
trap "exit 1" INT TERM HUP
launchctl bootout gui/$U/dev.aris.server 2>/dev/null && echo "$(stamp) Aris Server stopped for: speed_dspark $*" | tee -a $LOG
sleep 5
PYTHONPATH=$HOME/inkling-work/src ~/Projects/aris-server/.venv/bin/python bench/speed_dspark.py "$@" 2>&1 | tee -a $LOG
