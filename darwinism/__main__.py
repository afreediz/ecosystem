"""``python -m darwinism [run|live] ...`` -> the headless experiment (default) or live viewer.

Dispatches to ``darwinism.cli.experiment`` / ``darwinism.cli.live``. With no subcommand (or
``run``) it runs the headless experiment; ``live`` opens the Arcade viewer.
"""
from __future__ import annotations

import sys


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "live":
        sys.argv = [sys.argv[0], *argv[1:]]
        from darwinism.cli.live import main as _main
    else:
        if argv and argv[0] == "run":
            sys.argv = [sys.argv[0], *argv[1:]]
        from darwinism.cli.experiment import main as _main
    _main()


if __name__ == "__main__":
    main()
