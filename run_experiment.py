"""Back-compat shim: ``python run_experiment.py ...`` -> ``darwinism.cli.experiment``.

The framework's canonical entry points are the installed console script ``darwinism-run``
and ``python -m darwinism``. This shim keeps the historical invocation working from a bare
checkout (repo root is on sys.path, so ``darwinism`` imports without an install).
"""
from darwinism.cli.experiment import main

if __name__ == "__main__":
    main()
