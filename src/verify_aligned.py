import ast,importlib.util,json,math
import torch
import torch.nn.functional as F
from aligned_core import ROOT,HIST,source,angle_loss_2024,contrastive_eq4,objective,provenance
torch.manual_seed(2026)
x=torch.randn(8,768,requires_grad=True);scores=torch.tensor([1.,3.,4.,5.])
labels=scores.repeat_interleave(2).reshape(-1,1)
# Independently evaluate Eq4 using scalar logsumexp over selected positive pairs.
q=F.normalize(x[::2][scores>=3],dim=1);k=F.normalize(x[1::2][scores>=3],dim=1)
manual=torch.stack([torch.logsumexp(q[i]@k.T/.05,dim=0)-(q[i]*k[i]).sum()/.05 for i in range(3)]).mean()
impl=contrastive_eq4(x,scores,3)
assert torch.allclose(impl,manual,atol=1e-5)
g1=torch.autograd.grad(manual,x,retain_graph=True)[0];g2=torch.autograd.grad(impl,x,retain_graph=True)[0]
assert torch.allclose(g1,g2,atol=1e-5)
all_zero=contrastive_eq4(x,scores,6);assert all_zero.item()==0 and torch.isfinite(all_zero)
for mode in ['joint','cl_only','angle_only']:
    z,a,c=objective(x,scores,mode,10,3)
    assert torch.allclose(z,a+10*c)
    grad=torch.autograd.grad(z,x,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad)>0
assert objective(x,scores,'cl_only',10,3)[1].item()==0
assert objective(x,scores,'angle_only',10,3)[2].item()==0
spec=importlib.util.spec_from_file_location('modern',ROOT/'vendor-AnglE/angle_emb/loss.py')
modern=importlib.util.module_from_spec(spec);spec.loader.exec_module(modern)
old=angle_loss_2024(labels,x,20);new=modern.angle_loss(labels,x,20)
assert torch.equal(old,new),'Historical and modern angle differ: investigate'
# Audit historical target construction with continuous STS converted to integer.
legacy_labels=torch.tensor([1,1,3,3,2,2,1,1]).reshape(-1,1)
idx=torch.arange(8)
left=idx[None,:]*legacy_labels.T
right=(idx+1-idx%2*2)[:,None]*legacy_labels
targets=left==right
partner=torch.zeros((8,8),dtype=torch.bool)
for i in range(8):partner[i,i+1-i%2*2]=True
off_pair=(targets & ~partner).nonzero().tolist()
assert len(off_pair)>0
result=provenance()|{'eq4_scalar_and_gradient_parity':True,'empty_positive_batch_safe':True,
    'zero_cosine_objective':True,'ablation_gradients_finite':True,'angle_2024_equals_current_on_probe':True,
    'historical_integer_label_probe':[1,1,3,3,2,2,1,1],'historical_off_pair_targets':off_pair,
    'interpretation':'Historical STS README is not a complete unambiguous Table2 recipe; no claim of a paper implementation bug.'}
(ROOT/'provenance/alignment/verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
