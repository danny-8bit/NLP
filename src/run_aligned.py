"""Predeclared validation grid, freeze selection, five independent seeds, then test."""
import json,subprocess,sys,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'provenance/alignment';P.mkdir(parents=True,exist_ok=True)
R=ROOT/'runs_aligned';R.mkdir(exist_ok=True)
plan={'scope':'Table2 STS-B, BERT-base, paper Eq5 two-objective structure',
      'search_seed':42,'final_seeds':[142,143,144,145,146],
      'cl_weights':[1.,10.,20.],'positive_thresholds':[3.,4.],
      'angle_weight':1.,'cosine_weight':0.,'epochs':5,'batch_size':32,
      'selection':'maximum validation Spearman, deterministic grid order tie-break',
      'test_policy':'no test access in search; select/freeze before final runs; evaluate final tests after all training',
      'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if not (P/'plan.json').exists():(P/'plan.json').write_text(json.dumps(plan,indent=2))
def call(run_id,mode,seed,weight,threshold,test=False):
    cmd=[sys.executable,str(ROOT/'src/train_aligned.py'),'--run-id',run_id,'--mode',mode,'--seed',str(seed),'--cl-weight',str(weight),'--threshold',str(threshold)]
    if test:cmd.append('--test')
    with open(R/(run_id+('_test' if test else '')+'.log'),'a') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    print('FINISHED',run_id,'test' if test else 'train',flush=True)
if not (P/'selection.json').exists():
    scores=[]
    for t in plan['positive_thresholds']:
        for w in plan['cl_weights']:
            rid=f'search_t{int(t)}_w{int(w)}'
            call(rid,'joint',42,w,t)
            scores.append(json.loads((R/rid/'train_result.json').read_text()))
    selected=max(scores,key=lambda x:x['validation_spearman'])
    selection={**selected,'search_results':scores,'selection_time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'final_run_ids':[f'{m}_seed{s}' for s in plan['final_seeds'] for m in ['joint','cl_only','angle_only']]}
    (P/'selection.json').write_text(json.dumps(selection,indent=2))
selected=json.loads((P/'selection.json').read_text())
print('LOCKED SELECTION',selected['cl_weight'],selected['threshold'],flush=True)
for seed in plan['final_seeds']:
    for mode in ['joint','cl_only','angle_only']:
        call(f'{mode}_seed{seed}',mode,seed,selected['cl_weight'],selected['threshold'])
for seed in plan['final_seeds']:
    for mode in ['joint','cl_only','angle_only']:
        call(f'{mode}_seed{seed}',mode,seed,selected['cl_weight'],selected['threshold'],test=True)
print('ALL ALIGNED EXPERIMENTS COMPLETE',flush=True)
