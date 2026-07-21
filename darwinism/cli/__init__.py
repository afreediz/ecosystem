"""Command-line entry points for the darwinism framework.

``experiment`` -- headless, fast-forward run that writes a CSV (the reproducible path).
``live``       -- Arcade observer window (needs a display + the ``[render]`` extra).

Invoke via the installed console scripts ``darwinism-run`` / ``darwinism-live``, via
``python -m darwinism [run|live] ...``, or the modules directly
(``python -m darwinism.cli.experiment``).
"""
