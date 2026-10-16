"""Shared logging helpers for the stage scripts (any venv).

log()       one timestamped progress line to stdout (the short episode log)
to_debug()  send everything written to fd 1/2 — including subprocesses and C extensions —
            to the episode's debug log for the duration of the block
"""
from __future__ import annotations

import contextlib
import os
import sys
import time


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextlib.contextmanager
def to_debug(path: str | None, title: str = ""):
    if not path:
        yield
        return
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(f"\n===== {time.strftime('%H:%M:%S')} {title} =====\n"); fh.flush()
        os.dup2(fh.fileno(), 1); os.dup2(fh.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            os.close(saved[0]); os.close(saved[1])
