import json,urllib.request,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
url='https://huggingface.co/api/models/google-bert/bert-base-uncased?blobs=true'
with urllib.request.urlopen(url,timeout=60) as r: info=json.load(r)
entry=next(x for x in info['siblings'] if x['rfilename']=='model.safetensors')
expected=entry.get('lfs',{}).get('sha256')
assert expected, 'No official weight digest returned'
actual=hashlib.file_digest(open(ROOT/'models/bert-base-uncased/model.safetensors','rb'),'sha256').hexdigest()
assert expected==actual, 'Official BERT weight digest mismatch'
result={'official_model_revision':info['sha'],'official_lfs_sha256':expected,'local_sha256':actual,'match':True}
(ROOT/'provenance/official-weight-verification.json').write_text(json.dumps(result,indent=2))
print(result)
