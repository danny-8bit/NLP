#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python src/verify_aligned.py
.venv/bin/python src/run_aligned.py
.venv/bin/python src/report_aligned.py
.venv/bin/python src/package_aligned.py
