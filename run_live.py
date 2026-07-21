"""Back-compat shim: ``python run_live.py ...`` -> ``darwinism.cli.live``.

The framework's canonical entry point is the installed console script ``darwinism-live`` (or
``python -m darwinism live``). This shim keeps the historical invocation working from a bare
checkout (repo root is on sys.path, so ``darwinism`` imports without an install).
"""
from darwinism.cli.live import main

if __name__ == "__main__":
    main()
