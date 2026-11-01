"""AoE STS-B in-domain reproduction. See README for deviations from Table 2."""
import argparse, csv, hashlib, importlib.util, json, random, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('official_loss', ROOT/'vendor-AnglE/angle_emb/loss.py')
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_data(split):
    with open(ROOT/f'data/{split}.jsonl', encoding='utf8') as f:
        return [json.loads(line) for line in f]

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')

def evaluate(model, tokenizer, rows, batch_size=32):
    model.eval(); predictions=[]
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch=rows[start:start+batch_size]
            texts=[text for row in batch for text in (row['text1'],row['text2'])]
            enc=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                emb=model(**enc).last_hidden_state[:,0].float()
            pred=torch.nn.functional.cosine_similarity(emb[::2],emb[1::2]).cpu().numpy()
            predictions.extend(pred.tolist())
    gold=[row['score'] for row in rows]
    metrics={'n':len(rows),'spearman':float(spearmanr(gold,predictions).statistic),
             'pearson':float(pearsonr(gold,predictions).statistic)}
    return metrics,predictions

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['aoe','no_angle','cosent','untrained'],default='aoe')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    tag=f'{args.method}_seed{args.seed}'+('_smoke' if args.smoke else '')
    out=ROOT/'runs'/tag
    if (out/'metrics.json').exists():
        print('Already completed:',tag,flush=True); return
    out.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA unavailable'
    assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/bert-base-uncased',local_files_only=True)
    model=AutoModel.from_pretrained(ROOT/'models/bert-base-uncased',local_files_only=True).cuda()
    train,valid,test=[load_data(split) for split in ['train','validation','test']]
    if args.smoke:
        train,valid,test=train[:64],valid[:64],test[:64]
    config=vars(args)|{'model':'google-bert/bert-base-uncased','pooling':'last_hidden_state[:,0]',
        'max_length':128,'learning_rate':5e-5,'precision':'BF16 encoder; FP32 loss',
        'angle_weight':0.02 if args.method=='aoe' else 0.,'cosine_weight':1.,
        'ibn_weight':0. if args.method=='cosent' else 1.,'scale':20.,
        'ibn_positive_threshold':3.,'selection':'maximum validation Spearman per epoch',
        'paper_reference':'Table 2, STS-B column; not bitwise historical reproduction'}
    dump(out/'config.json',config)
    def collate(batch):
        texts=[text for row in batch for text in (row['text1'],row['text2'])]
        return tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='pt'),torch.tensor([r['score'] for r in batch])
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=collate,
                      generator=torch.Generator().manual_seed(args.seed),num_workers=0)
    epochs=1 if args.smoke else args.epochs
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.)
    scheduler=get_linear_schedule_with_warmup(optimizer,0,epochs*len(loader))
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
    history=[]; best=-float('inf'); checkpoint=out/'best.pt'
    if args.method!='untrained':
        for epoch in range(epochs):
            model.train(); total=0.
            for step,(enc,score) in enumerate(loader):
                enc={k:v.cuda() for k,v in enc.items()}; score=score.cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    emb=model(**enc).last_hidden_state[:,0].float()
                # Official losses require alternating sentence pairs and repeated labels.
                ranking=score.repeat_interleave(2).reshape(-1,1)
                loss=losses.cosine_loss(ranking,emb,tau=20.)
                if args.method!='cosent':
                    # Explicit adaptation: continuous STS labels cannot be cast to int
                    # inside the official binary contrastive target construction.
                    binary=(score>=3.).float().repeat_interleave(2).reshape(-1,1)
                    loss=loss+losses.in_batch_negative_loss(binary,emb,tau=20.)
                if args.method=='aoe':
                    loss=loss+.02*losses.angle_loss(ranking,emb,tau=20.)
                assert torch.isfinite(loss),f'Nonfinite loss at {epoch}/{step}'
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); scheduler.step(); total+=loss.item()
                if step%50==0: print(tag,epoch+1,step,round(loss.item(),5),flush=True)
            metrics,_=evaluate(model,tokenizer,valid)
            history.append({'epoch':epoch+1,'loss':total/len(loader),**metrics})
            dump(out/'history.json',history)
            print('Validation:',history[-1],flush=True)
            if metrics['spearman']>best:
                best=metrics['spearman']; torch.save(model.state_dict(),checkpoint)
        model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    # Test set is evaluated only after checkpoint selection using validation.
    metrics,pred=evaluate(model,tokenizer,test)
    torch.cuda.synchronize()
    metrics|={'method':args.method,'seed':args.seed,'selected_validation_spearman':best if history else None,
              'selected_epoch':max(history,key=lambda x:x['spearman'])['epoch'] if history else 0,
              'seconds':time.perf_counter()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3}
    with open(out/'test_predictions.csv','w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['id','text1','text2','score','prediction']);writer.writeheader()
        for row,p in zip(test,pred):writer.writerow(row|{'prediction':p})
    if args.method=='aoe' and args.seed==42 and not args.smoke:
        model.save_pretrained(out/'model');tokenizer.save_pretrained(out/'model')
    dump(out/'metrics.json',metrics)
    if checkpoint.exists():checkpoint.unlink()
    print('COMPLETE',tag,metrics,flush=True)

if __name__=='__main__':main()
