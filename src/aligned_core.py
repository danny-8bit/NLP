"""Paper Eq.(4)-(5) structure plus exactly extracted 2024 official angle function."""
import ast, hashlib, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
COMMIT='e10d5511a76e6a4b1cc55fe67969cf9afb28d0e2'
HIST=ROOT/f'provenance/alignment/AnglE-{COMMIT}/angle_emb/angle.py'
source=HIST.read_text(encoding='utf8')
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='angle_loss')
angle_source=ast.get_source_segment(source,node)
namespace={'torch':torch,'F':F}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(HIST),'exec'),namespace)
angle_loss_2024=namespace['angle_loss']

def contrastive_eq4(emb, scores, positive_threshold, temperature=.05):
    """Positive pairs only; other positive targets are in-batch negatives.

    Eq.4 summed anchor losses reduced to mean; coefficient scale is validation-selected.
    STS positive threshold is an explicitly under-specified reconstruction parameter.
    """
    mask=scores>=positive_threshold
    q=F.normalize(emb[::2][mask],dim=-1)
    k=F.normalize(emb[1::2][mask],dim=-1)
    if q.shape[0]<2:return emb.sum()*0
    logits=q@k.T/temperature
    return F.cross_entropy(logits,torch.arange(len(q),device=emb.device))

def objective(emb,scores,mode,cl_weight,threshold):
    labels=scores.repeat_interleave(2).reshape(-1,1)
    a=angle_loss_2024(labels,emb,tau=20.) if mode!='cl_only' else emb.sum()*0
    c=contrastive_eq4(emb,scores,threshold) if mode!='angle_only' else emb.sum()*0
    return a+cl_weight*c, a, c

def provenance():
    return {'historical_commit':COMMIT,'angle_function_sha256':hashlib.sha256(angle_source.encode()).hexdigest(),
            'historical_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'source_lines':[node.lineno,node.end_lineno],
            'cosine_weight':0,'angle_weight':1,'temperature':.05,
            'cl_reduction':'mean over positive anchors; positive target candidates only',
            'label_policy':'continuous STS scores for angle; validation-selected threshold for CL; not asserted as original Table2 setting'}
