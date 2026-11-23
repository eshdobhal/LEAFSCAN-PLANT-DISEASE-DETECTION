"""One command to check the whole repository: build, then every test suite in the environment it belongs to.

    python scripts/check_all.py            # build check + fundamentals, surveillance, web app, HTML checks
    python scripts/check_all.py --e2e      # also the 15 web-app tests in a real Chromium browser
    python scripts/check_all.py --site _site   # also the browser version (built by web/build_site.py)

LeafScan has two environments: the web app's (`.venv`, CPU PyTorch + Flask) and the research one
(`.venv-train`, GPU PyTorch + detection / segmentation libraries). Each suite runs with the Python of its
environment; if that environment does not exist, the Python running this script is used.
Exit code 0 only if every step passes. A summary table is printed at the end.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "Flask Deployed App"


def env_python(name):
    for rel in ("Scripts/python.exe", "bin/python"):
        p = ROOT / name / rel
        if p.exists():
            return str(p)
    return sys.executable


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e2e", action="store_true", help="include the browser tests of the web app")
    ap.add_argument("--site", type=Path, help="also test a built browser version in this folder")
    args = ap.parse_args(argv)
    research, web = env_python(".venv-train"), env_python(".venv")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")  # tests run on the CPU
    steps = [
        ("build: package metadata and imports", research, ["-c", "import fundamentals, surveillance.cli, surveillance.pipeline; print('imports ok')"], ROOT, {}),
        ("build: byte-compile all sources", research, ["-m", "compileall", "-q", "fundamentals", "surveillance", "training", "web", "scripts", "demo"], ROOT, {}),
        ("fundamentals (NumPy)", research, ["-m", "pytest", "-q", "fundamentals/tests"], ROOT, {}),
        ("surveillance", research, ["-m", "pytest", "-q", "surveillance/tests"], ROOT, {}),
        ("web app" + (" incl. browser" if args.e2e else ""), web, ["-m", "pytest", "-q"], APP, {"E2E": "1"} if args.e2e else {}),
        ("HTML checks", web, ["scripts/check_html.py"], ROOT, {}),
    ]
    if args.site:
        steps.append(("browser version", web, ["web/test_web.py", str(args.site)], ROOT, {}))
    results = []
    for name, python, cmd, cwd, extra in steps:
        print(f"\n=== {name}: {Path(python).parent.parent.name} {' '.join(cmd)}", flush=True)
        t = time.perf_counter()
        r = subprocess.run([python, *cmd], cwd=cwd, env={**env, **extra}, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (r.stdout + r.stderr).strip().splitlines()
        print("\n".join(out[-6:]))
        results.append((name, r.returncode == 0, out[-1] if out else "", time.perf_counter() - t))
    print("\n" + "-" * 92)
    for name, ok, last, secs in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:42s} {secs:6.1f}s  {last[:36]}")
    failed = sum(not ok for _, ok, _, _ in results)
    print("-" * 92 + f"\n{'ALL PASSED' if not failed else f'{failed} STEP(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
