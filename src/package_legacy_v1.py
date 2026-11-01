import hashlib, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert (ROOT/'report/report.md').exists(), 'Final report missing'
output=ROOT/'deliverables';output.mkdir(exist_ok=True)
archive=output/'AoE_midterm.zip'
folders=['src','scripts','data','models','runs','results','report','provenance','vendor-AnglE']
files=[ROOT/'README.md']
for folder in folders:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p) and '_smoke' not in str(p) and '.git/' not in str(p) and p.suffix!='.pt')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in sorted(files):z.write(p,Path('AoE_midterm')/p.relative_to(ROOT))
with zipfile.ZipFile(archive) as z:
    bad=z.testzip();assert bad is None,bad
sha=hashlib.file_digest(open(archive,'rb'),'sha256').hexdigest()
(output/'SHA256.txt').write_text(sha+'  '+archive.name+'\n')
print('Archive verified:',archive,archive.stat().st_size,sha)
