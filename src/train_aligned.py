import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from aligned_core import ROOT,objective,provenance
from experiment import evaluate,load_data,dump,seed_all

def train(args):
    out=ROOT/'runs_aligned'/args.run_id
    if (out/'train_result.json').exists():return json.loads((out/'train_result.json').read_text())
    if out.exists() and any(out.iterdir()):raise RuntimeError(f'Incomplete run exists; preserve and rename before rerun: {out}')
    out.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed);torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/bert-base-uncased',local_files_only=True)
    model=AutoModel.from_pretrained(ROOT/'models/bert-base-uncased',local_files_only=True).cuda()
    tr,va=load_data('train'),load_data('validation')
    if args.smoke:tr,va=tr[:64],va[:64]
    config=vars(args)|provenance()|{'max_length':128,'pooling':'CLS','learning_rate':5e-5,
        'batch_size':32,'weight_decay':0,'warmup_steps':0,'precision':'BF16 encoder, FP32 loss',
        'test_access_during_train':False,'selection':'validation Spearman only'}
    dump(out/'config.json',config)
    def collate(rows):
        return tokenizer([s for r in rows for s in [r['text1'],r['text2']]],padding=True,truncation=True,max_length=128,return_tensors='pt'),torch.tensor([r['score'] for r in rows])
    loader=DataLoader(tr,batch_size=32,shuffle=True,collate_fn=collate,num_workers=0,generator=torch.Generator().manual_seed(args.seed))
    epochs=1 if args.smoke else 5
    optim=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0)
    sched=get_linear_schedule_with_warmup(optim,0,len(loader)*epochs)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    history=[];best=-float('inf')
    for epoch in range(epochs):
        model.train();totals=np.zeros(3)
        for step,(enc,scores) in enumerate(loader):
            enc={k:v.cuda() for k,v in enc.items()};scores=scores.cuda();optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):emb=model(**enc).last_hidden_state[:,0].float()
            loss,a,c=objective(emb,scores,args.mode,args.cl_weight,args.threshold)
            assert torch.isfinite(loss),f'{args.run_id}: nonfinite loss'
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step();sched.step()
            totals+=np.array([loss.item(),a.item(),c.item()])
            if step%60==0:print(args.run_id,epoch+1,step,loss.item(),flush=True)
        metrics,_=evaluate(model,tokenizer,va)
        row={'epoch':epoch+1,'loss':totals[0]/len(loader),'angle_loss':totals[1]/len(loader),'cl_loss':totals[2]/len(loader),**metrics}
        history.append(row);dump(out/'history.json',history);print('VALIDATION',args.run_id,row,flush=True)
        if row['spearman']>best:
            best=row['spearman'];torch.save(model.state_dict(),out/'best.pt')
    torch.cuda.synchronize()
    result={'run_id':args.run_id,'seed':args.seed,'mode':args.mode,'cl_weight':args.cl_weight,'threshold':args.threshold,
            'validation_spearman':best,'selected_epoch':max(history,key=lambda x:x['spearman'])['epoch'],
            'seconds':time.perf_counter()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3}
    dump(out/'train_result.json',result)
    return result

def test(args):
    assert not args.smoke,'Smoke never evaluates test'
    out=ROOT/'runs_aligned'/args.run_id
    if (out/'metrics.json').exists():return
    locked=json.loads((ROOT/'provenance/alignment/selection.json').read_text())
    assert args.run_id in locked['final_run_ids'],'Test access limited to predeclared final runs'
    cfg=json.loads((out/'config.json').read_text())
    assert cfg['threshold']==locked['threshold'] and cfg['cl_weight']==locked['cl_weight']
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/bert-base-uncased',local_files_only=True)
    model=AutoModel.from_pretrained(ROOT/'models/bert-base-uncased',local_files_only=True).cuda()
    model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=True))
    rows=load_data('test');metrics,preds=evaluate(model,tokenizer,rows)
    metrics|=json.loads((out/'train_result.json').read_text())
    with open(out/'test_predictions.csv','w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['id','text1','text2','score','prediction']);writer.writeheader()
        for row,p in zip(rows,preds):writer.writerow(row|{'prediction':p})
    if args.mode=='joint' and args.seed==142:
        model.save_pretrained(out/'model');tokenizer.save_pretrained(out/'model')
    dump(out/'metrics.json',metrics)
    (out/'best.pt').unlink()
    print('TEST',metrics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--mode',choices=['joint','cl_only','angle_only'],default='joint')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--cl-weight',type=float,default=20.)
    p.add_argument('--threshold',type=float,default=3.);p.add_argument('--smoke',action='store_true')
    p.add_argument('--test',action='store_true');args=p.parse_args()
    test(args) if args.test else train(args)
