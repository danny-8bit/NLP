import json, hashlib, urllib.request, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def get(url):
    with urllib.request.urlopen(url,timeout=90) as r:return r.read()

model=ROOT/'models/bert-base-uncased';model.mkdir(parents=True,exist_ok=True)
info=json.loads(get('https://huggingface.co/api/models/google-bert/bert-base-uncased'))
rev=info['sha']
for name in ['config.json','tokenizer.json','tokenizer_config.json','vocab.txt']:
    target=model/name
    if not target.exists():target.write_bytes(get(f'https://huggingface.co/google-bert/bert-base-uncased/resolve/{rev}/{name}'))
if not (model/'model.safetensors').exists():
    urllib.request.urlretrieve(f'https://huggingface.co/google-bert/bert-base-uncased/resolve/{rev}/model.safetensors',model/'model.safetensors')
from datasets import load_dataset
meta=json.loads(get('https://huggingface.co/api/datasets/mteb/stsbenchmark-sts'))
data=load_dataset('mteb/stsbenchmark-sts',revision=meta['sha'])
data.save_to_disk(str(ROOT/'data/raw_stsb'))
sizes={}
for split in ['train','validation','test']:
    sizes[split]=len(data[split])
    with open(ROOT/f'data/{split}.jsonl','w',encoding='utf8') as f:
        for i,row in enumerate(data[split]):
            score=float(row['score']);assert 0<=score<=5
            f.write(json.dumps({'id':f'{split}-{i}','text1':row['sentence1'],'text2':row['sentence2'],'score':score},ensure_ascii=False)+'\n')
provenance={'model_revision':rev,'dataset_revision':meta['sha'],'sizes':sizes,'sha256':{}}
for path in list(model.glob('*'))+list((ROOT/'data').glob('*.jsonl'))+[ROOT/'vendor-AnglE/angle_emb/loss.py']:
    provenance['sha256'][str(path.relative_to(ROOT))]=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()
(ROOT/'provenance/assets.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
print(provenance,flush=True)
