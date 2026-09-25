# Remote probe, piped through `colab exec -f`. Prints ONE line the watcher parses:
#   REMOTE_OK <tag> <rows> <alive:0|1> <last log line>
# The REMOTE_OK sentinel is the whole point (Monitor v3, RUNLOG): its absence is a
# client-side flake, not a dead run. Three consecutive misses = presumed VM loss.
# __TAG__ / __OUT__ / __LOG__ / __PROC__ are template slots: neither argv nor the local
# environment crosses `colab exec` (found 2026-09-03, first probe missed), so watch.sh
# seds a per-tag copy.
import os, subprocess
tag = "__TAG__"
out = "__OUT__"
log = "__LOG__"
rows = sum(1 for _ in open(out)) if os.path.exists(out) else 0
alive = subprocess.run(["pgrep", "-f", "__PROC__"], capture_output=True).returncode == 0
last = ""
if os.path.exists(log):
    lines = open(log, errors="replace").read().strip().splitlines()
    last = lines[-1][-160:] if lines else ""
print(f"REMOTE_OK {tag} {rows} {int(alive)} {last}")
