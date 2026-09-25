#!/usr/bin/env python
"""Run one `colab` CLI command with a HARD client-side timeout. The CLI's --timeout bounds
remote execution only; a hung client call (2026-09-04: watch.sh froze 2.5 h inside one exec,
machine alive the whole time) has no bound at all, and macOS ships no `timeout`.
Usage: colab_call.py <seconds> <colab args...>     exit 124 on timeout."""
import subprocess, sys
from pathlib import Path
secs = float(sys.argv[1]); args = sys.argv[2:]
colab = str(Path(__file__).resolve().parent.parent / ".venv/bin/colab")
try:
    r = subprocess.run([colab, *args], capture_output=True, text=True, timeout=secs)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)
except subprocess.TimeoutExpired:
    sys.stderr.write(f"colab_call: timeout after {secs}s: {' '.join(args)}\n"); sys.exit(124)
