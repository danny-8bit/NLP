import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sel=json.loads((ROOT/'provenance/alignment/selection.json').read_text())
for rid in sel['final_run_ids']:assert (ROOT/f'runs_aligned/{rid}/metrics.json').exists(),rid
assert (ROOT/'report/report.html').exists()
output=ROOT/'deliverables';output.mkdir(exist_ok=True)
archive=output/'AoE_midterm.zip'
folders=['src','scripts','data','models','runs','runs_aligned','results','results_aligned','report','provenance','vendor-AnglE']
files=[ROOT/'README.md']
for folder in folders:
    for p in (ROOT/folder).rglob('*'):
        rel=p.relative_to(ROOT)
        if not p.is_file() or '__pycache__' in p.parts or '.git' in p.parts or p.suffix=='.pt':continue
        if rel.parts[:3]==('runs','aoe_seed42','model'):continue
        files.append(p)
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in sorted(files):z.write(p,Path('AoE_midterm')/p.relative_to(ROOT))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=set(z.namelist())
    for rid in sel['final_run_ids']:assert f'AoE_midterm/runs_aligned/{rid}/test_predictions.csv' in names
sha=hashlib.file_digest(open(archive,'rb'),'sha256').hexdigest()
(output/'SHA256.txt').write_text(sha+'  '+archive.name+'\n')
print('Verified archive:',archive,archive.stat().st_size,sha)
