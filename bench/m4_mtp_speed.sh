#!/bin/zsh
# The MTP speed run on the M4, from ~/mtp-work (a copy of this worktree's src/ and bench/, plus the head file).
# Refuses while another GPU job runs; stops Aris Server only while idle; the EXIT trap always restores it.
#   zsh bench/m4_mtp_speed.sh --model ~/models/arete-27b-expert1-mlx-8bit --mtp ~/mtp-work/mtp-raw.safetensors \
#       --prompts bench/prompts.json --out runs/mtp-speed-greedy.json
set -u
cd ~/mtp-work || exit 9
U=$(id -u); PLIST=~/Library/LaunchAgents/dev.aris.server.plist
mkdir -p runs
LOG=runs/mtp-speed.log
stamp() { date '+%F %T'; }
busy() { [ "$(lsof -nP -iTCP:8484 -sTCP:ESTABLISHED 2>/dev/null | grep -c ESTABLISHED)" -gt 0 ]; }
others() { pgrep -fl 'oMLX|omlx|opencode|run_native_trial|hard_pilot|agent_loop|speed_mtp' | grep -v -e pgrep -e "$$"; }
if [ -n "$(others)" ]; then echo "$(stamp) another GPU job is running, not starting: $(others | head -3 | tr '\n' ';')" | tee -a $LOG; exit 3; fi
if busy; then echo "$(stamp) Aris Server is busy, not stopping it" | tee -a $LOG; exit 2; fi
restore() {
  launchctl bootstrap gui/$U $PLIST 2>/dev/null; launchctl kickstart gui/$U/dev.aris.server 2>/dev/null
  echo "$(stamp) Aris Server restored" | tee -a $LOG
}
trap restore EXIT
trap "exit 1" INT TERM HUP
launchctl bootout gui/$U/dev.aris.server 2>/dev/null && echo "$(stamp) Aris Server stopped for: speed_mtp $*" | tee -a $LOG
sleep 5
PYTHONPATH=$HOME/mtp-work/src ~/Projects/aris-server/.venv/bin/python bench/speed_mtp.py "$@" 2>&1 | tee -a $LOG
