"""Independent sanity checks for official loss and experiment inputs."""
import importlib.util,json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('official',ROOT/'vendor-AnglE/angle_emb/loss.py')
loss=importlib.util.module_from_spec(spec);spec.loader.exec_module(loss)
torch.manual_seed(42)
x=torch.randn(8,768,requires_grad=True)
y=torch.tensor([0.,0.,1.,1.,3.,3.,5.,5.]).reshape(-1,1)
binary=(y>=3).float()
parts=[loss.cosine_loss(y,x,20),loss.in_batch_negative_loss(binary,x,20),loss.angle_loss(y,x,20)]
total=parts[0]+parts[1]+.02*parts[2];total.backward()
assert torch.isfinite(total) and torch.isfinite(x.grad).all()
assert torch.count_nonzero(x.grad)>0
scaled=loss.angle_loss(y,x.detach()*3.,20)
assert torch.allclose(parts[2].detach(),scaled,atol=1e-4), 'Angle normalization scale invariance failed'
splits={}
for split in ['train','validation','test']:
    rows=[json.loads(x) for x in (ROOT/f'data/{split}.jsonl').read_text().splitlines()]
    assert all(0<=r['score']<=5 for r in rows)
    splits[split]={tuple(sorted((r['text1'],r['text2']))) for r in rows}
overlap={f'{a}/{b}':len(splits[a]&splits[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
assets=json.loads((ROOT/'provenance/assets.json').read_text())
for relative,expected in assets['sha256'].items():
    actual=hashlib.file_digest(open(ROOT/relative,'rb'),'sha256').hexdigest()
    assert actual==expected,relative
result={'loss_components':[float(p.detach()) for p in parts],'finite_nonzero_gradient':True,
        'scale_invariance':True,'hashes_verified':len(assets['sha256']),
        'exact_unordered_pair_overlap':overlap,
        'overlap_policy':'Unmodified official dataset split; report any intrinsic overlap, do not silently filter test data.'}
(ROOT/'provenance/verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
