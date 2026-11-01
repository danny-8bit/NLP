"""Finalize only after all predeclared aligned experiments have test evidence."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
selection=json.loads((ROOT/'provenance/alignment/selection.json').read_text())
for rid in selection['final_run_ids']:
    assert (ROOT/f'runs_aligned/{rid}/metrics.json').exists(),f'Missing test run {rid}'
legacy=ROOT/'report/legacy_v1'
if not legacy.exists():
    legacy.mkdir()
    for name in ['report.md','report.html']:
        if (ROOT/'report'/name).exists():shutil.copy2(ROOT/'report'/name,legacy/name)
    if (legacy/'report.md').exists():
        text=(legacy/'report.md').read_text().replace('../results/','../../results/')
        (legacy/'report.md').write_text(text)
old_readme=ROOT/'provenance/README_v1.md'
if not old_readme.exists():shutil.copy2(ROOT/'README.md',old_readme)
if not (ROOT/'src/report_legacy_v1.py').exists():
    shutil.copy2(ROOT/'src/report.py',ROOT/'src/report_legacy_v1.py')
(ROOT/'src/report.py').write_text("from pathlib import Path\nimport runpy\nrunpy.run_path(str(Path(__file__).with_name('report_aligned.py')), run_name='__main__')\n")
if not (ROOT/'src/package_legacy_v1.py').exists():
    shutil.copy2(ROOT/'src/package.py',ROOT/'src/package_legacy_v1.py')
(ROOT/'src/package.py').write_text("from pathlib import Path\nimport runpy\nrunpy.run_path(str(Path(__file__).with_name('package_aligned.py')), run_name='__main__')\n")
(ROOT/'scripts/run_all.sh').write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/..\"\n.venv/bin/python src/verify_aligned.py\n.venv/bin/python src/run_aligned.py\n.venv/bin/python src/report_aligned.py\n.venv/bin/python src/package_aligned.py\n")
stage=Path('/mnt/c/Users/cdy/Documents/Codex/2026-10-04/bang/work/aoe_stage')
shutil.copy2(stage/'README_aligned.md',ROOT/'README.md')
subprocess.run([sys.executable,str(ROOT/'src/report_aligned.py')],check=True,cwd=ROOT)
manifest={}
for folder in ['src','scripts','data','runs_aligned','results_aligned','report']:
    for p in (ROOT/folder).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pt':continue
        manifest[str(p.relative_to(ROOT))]=hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
(ROOT/'provenance/alignment/submission_manifest.json').write_text(json.dumps(manifest,indent=2))
subprocess.run([sys.executable,str(ROOT/'src/package_aligned.py')],check=True,cwd=ROOT)
output=Path('/mnt/c/Users/cdy/Documents/Codex/2026-10-04/bang/outputs');output.mkdir(exist_ok=True)
for source in ['deliverables/AoE_midterm.zip','deliverables/SHA256.txt','report/report.html','README.md',
               'results_aligned/summary.csv','results_aligned/comparison.png','results_aligned/validation_curves.png']:
    shutil.copy2(ROOT/source,output/Path(source).name)
(output/'report.md').write_text((ROOT/'report/report.md').read_text().replace('../results_aligned/','./'),encoding='utf8')
expected=(ROOT/'deliverables/SHA256.txt').read_text().split()[0]
actual=hashlib.file_digest(open(output/'AoE_midterm.zip','rb'),'sha256').hexdigest()
assert actual==expected
print('FINAL EXPORT VERIFIED',actual,flush=True)
