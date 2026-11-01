"""Final report for paper-structure-aligned v2; never substitutes old results."""
import base64,hashlib,json,re
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'runs_aligned';O=ROOT/'results_aligned';O.mkdir(exist_ok=True)
P=ROOT/'provenance/alignment'
plan=json.loads((P/'plan.json').read_text());selection=json.loads((P/'selection.json').read_text())
verification=json.loads((P/'verification.json').read_text());assets=json.loads((ROOT/'provenance/assets.json').read_text())
rows=[];frames={}
for rid in selection['final_run_ids']:
    row=json.loads((R/rid/'metrics.json').read_text());cfg=json.loads((R/rid/'config.json').read_text())
    assert cfg['cosine_weight']==0 and cfg['historical_commit']==verification['historical_commit']
    assert cfg['cl_weight']==selection['cl_weight'] and cfg['threshold']==selection['threshold']
    frames[rid]=pd.read_csv(R/rid/'test_predictions.csv');rows.append(row)
df=pd.DataFrame(rows).sort_values(['mode','seed']);df.to_csv(O/'all_runs.csv',index=False)
summary=df.groupby('mode').agg(n=('seed','count'),spearman_mean=('spearman','mean'),spearman_std=('spearman','std'),
    pearson_mean=('pearson','mean'),train_validation_seconds=('seconds','mean'),peak_memory_gib=('peak_memory_gib','max'))
summary.to_csv(O/'summary.csv');pd.DataFrame(selection['search_results']).to_csv(O/'validation_search.csv',index=False)
first=frames['joint_seed142'];gold=first.score.to_numpy()
for f in frames.values():assert f.id.tolist()==first.id.tolist() and np.array_equal(f.score.to_numpy(),gold)
preds={m:np.array([frames[f'{m}_seed{s}'].prediction.to_numpy() for s in plan['final_seeds']]) for m in ['joint','cl_only','angle_only']}
bootstrap={};rng=np.random.default_rng(2026)
for baseline in ['cl_only','angle_only']:
    differences=[]
    for _ in range(1000):
        ix=rng.integers(0,len(gold),len(gold))
        differences.append(np.mean([spearmanr(gold[ix],a[ix]).statistic-spearmanr(gold[ix],b[ix]).statistic for a,b in zip(preds['joint'],preds[baseline])]))
    bootstrap[baseline]={'mean_difference_points':float((summary.loc['joint','spearman_mean']-summary.loc[baseline,'spearman_mean'])*100),
        'conditional_test_item_95ci':(np.quantile(differences,[.025,.975])*100).tolist()}
(O/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2))
train=[json.loads(x) for x in (ROOT/'data/train.jsonl').read_text().splitlines()]
train_pairs={tuple(sorted([r['text1'],r['text2']])) for r in train}
mask=np.array([tuple(sorted([a,b])) not in train_pairs for a,b in zip(first.text1,first.text2)])
clean=[]
for rid,f in frames.items():clean.append({'run_id':rid,'n':int(mask.sum()),'spearman':float(spearmanr(f.score[mask],f.prediction[mask]).statistic)})
pd.DataFrame(clean).to_csv(O/'nonoverlap_sensitivity.csv',index=False)
first=first.copy();first['rank_error']=abs(rankdata(first.score)/len(first)-rankdata(first.prediction)/len(first))
first.sort_values('rank_error',ascending=False).head(20).to_csv(O/'largest_rank_errors.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4.5))
names=['joint','cl_only','angle_only'];means=summary.loc[names,'spearman_mean'].values*100;std=summary.loc[names,'spearman_std'].values*100
ax.bar(names,means,yerr=std,capsize=5,color=['#236c93','#6b8999','#9a735e'])
ax.set_ylabel('Test Spearman x 100');ax.set_title('Paper-structure-aligned v2: 5 seeds, sample SD')
for i,m in enumerate(means):ax.text(i,m+std[i]+1,f'{m:.2f}',ha='center')
ax.set_ylim(0,100);fig.tight_layout();fig.savefig(O/'comparison.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5))
for mode in names:
    h=json.loads((R/f'{mode}_seed142/history.json').read_text());ax.plot([r['epoch'] for r in h],[r['spearman']*100 for r in h],marker='o',label=mode)
ax.legend();ax.set_xlabel('Epoch');ax.set_ylabel('Validation Spearman x 100');ax.set_title('Validation curves, seed 142 only');fig.tight_layout();fig.savefig(O/'validation_curves.png',dpi=180);plt.close(fig)

def table(headers,body):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in body])
main_table=table(['方法','次数','Spearman×100（均值±样本SD）','Pearson×100','训练及验证秒/次','峰值显存GiB'],[
    [m,int(summary.loc[m,'n']),f"{summary.loc[m,'spearman_mean']*100:.2f}±{summary.loc[m,'spearman_std']*100:.2f}",f"{summary.loc[m,'pearson_mean']*100:.2f}",f"{summary.loc[m,'train_validation_seconds']:.1f}",f"{summary.loc[m,'peak_memory_gib']:.2f}"] for m in names])
search_table=table(['搜索编号','对比权重w₂','正例阈值t','最佳验证Spearman×100','选定epoch'],[[r['run_id'],r['cl_weight'],r['threshold'],f"{r['validation_spearman']*100:.3f}",r['selected_epoch']] for r in selection['search_results']])
seed_table=table(['种子','joint','cl_only','angle_only','joint−cl_only'],[[s]+[f"{df[(df['mode']==m)&(df.seed==s)].iloc[0].spearman*100:.3f}" for m in names]+[f"{(df[(df['mode']=='joint')&(df.seed==s)].iloc[0].spearman-df[(df['mode']=='cl_only')&(df.seed==s)].iloc[0].spearman)*100:+.3f}"] for s in plan['final_seeds']])
current=summary.loc['joint','spearman_mean']*100;delta=bootstrap['cl_only']['mean_difference_points'];ci=bootstrap['cl_only']['conditional_test_item_95ci']
conclusion='当前设置观察到正向差异，但外推需谨慎。' if ci[0]>0 else ('当前设置联合目标低于仅对比目标，不能声称复现了增益。' if ci[1]<0 else '条件区间包含0，当前设置未提供稳定增益证据。')
commit=verification['historical_commit']
report=f'''# AoE期中复现报告：双目标方法对齐与STS-B域内实验

成员甲：[姓名/学号待填写]；成员乙：[姓名/学号待填写]。本版替代旧适配版为主要结果，旧结果在runs/、results/和report/legacy_v1/保留。

## 摘要
围绕ACL2024论文AoE的Table 2 STS-B域内任务，本项目重新核查2024年作者源码，去除旧版额外cosine损失，对齐论文式(5)的“角度＋对比”双目标，并按式(4)重建正例配对的对比计算。先以种子42进行6组验证集搜索，锁定权重与正例划分，再以142–146五个种子分别训练联合、仅对比和仅角度目标，完成15组正式实验。联合目标测试Spearman×100为{current:.2f}±{summary.loc['joint','spearman_std']*100:.2f}，相对仅对比变化{delta:+.2f}个百分点。{conclusion}这是核心目标结构对齐的复现，正例划分、损失归约与Table2专项超参数仍有未公开细节，不能声称严格历史重现。

## 1 范围与待检验结论
论文：[1] Xianming Li and Jing Li. 2024. *AoE: Angle-optimized Embeddings for Semantic Textual Similarity*. Proceedings of ACL 2024 (Volume 1: Long Papers), 1825–1839. DOI:10.18653/v1/2024.acl-long.101。ACL主会属CCF人工智能A类。

| 论文证据 | 本项目范围 | 对应评价 |
|---|---|---|
| §3、Eq.2–5：角度目标配合对比目标 | BERT最后层CLS，历史官方角度函数与Eq.4对比项 | 数值/梯度测试与双目标消融 |
| §5.1、Table2：域内STS-B，AoE 86.28±0.19 | STS-B训练、验证选模、测试，五种子 | Spearman×100及Pearson |
| §5.2、Table4：角度与对比组件作用 | 在本登记的STS-B域内范围内补做组件消融 | 不把本次单数据集消融当成Table4七数据集均值复现 |
| GIS、NLI大规模训练、LLaMA、MTEB | 不在本次范围 | 不报告这些任务结论 |

登记状态：上述为项目范围说明，未提供教师确认记录，提交前应与实际登记一致。期中检验真实复现与分析，不要求必然优于论文或基线。

## 2 历史证据审计与实现修正
查询ACL2024发表前的作者提交，固定到2024-07-31的`{commit}`，完整源码和查询返回保存在provenance/alignment。该版本提供了双目标BERT示例（cosine_w=0），AngleLoss默认angle_w=1、ibn_w=20；但README的STS-B微调示例又使用NLI预训练模型、三损失及不同学习率，不能确认就是Table2原配置。

历史tokenizer把label转换为int，Format A文档却规定标签0/1，而STS示例直接传0–5分数。独立反例发现，整数标签`[1,1,3,3,2,2,1,1]`代入历史IBN目标构造，会产生句对以外的目标索引`{verification['historical_off_pair_targets']}`。这证明该示例存在接口歧义，不证明论文实验运行了该行为，更不据此断言论文结果错误。因此本次不照搬该标签构造，而用论文Eq.4显式构造正例对比。

| 项目 | 旧适配版 | 本次v2 | 对齐依据/剩余边界 |
|---|---|---|---|
| 总目标 | L_cos + L_ibn + 0.02 L_angle | L_angle + w₂ L_cl | Eq.5双目标；cosine权重严格0 |
| 角度函数 | 当前公开源码 | 从2024提交AST提取原函数，未经改写 | 历史angle.py第113–159行，保存函数哈希 |
| 排序标签 | 连续STS分数 | 保留连续0–5分数，不截断 | 符合Eq.3按相似度排序的目标；不复刻示例整数化 |
| 对比目标 | 二值标签送入官方2B矩阵 | 正例q/k相似度矩阵，交叉熵对角目标 | 对齐Eq.4；正例划分阈值仍是重建参数 |
| 权重与正例阈值 | 固定0.02及3 | 只在验证集搜索w₂∈{{1,10,20}}、t∈{{3,4}} | 原文说权重网格搜索，未公开STS具体最优值和正例划分 |
| 初始化 | 原始BERT-base | 同一官方原始BERT-base | 论文写BERT-base；README的NLI模型初始化不当作Table2证据 |
| 原有结果 | 12组适配实验 | 另存为历史参照 | 不把新旧多项修改的差值归因于单一组件 |

## 3 核心方法与代码
输入句对交错排列，取BERT最后层CLS的768维向量；历史角度函数将其划分为384维实部和384维虚部，计算归一化复数关系与排序损失。本次固定角度权重w₁=1，原始标注用于排序。

对比项先在当前批次按s≥t选出m个正例句对，令q为第一句、k为第二句，计算`logits[i,j]=cos(q_i,k_j)/0.05`，目标为对角配对i=j，其他正例对的第二句作为批内负例。`L_cl=mean_i(logsumexp_j(logits[i,j])-logits[i,i])`。论文写求和，本实现对有效正例取平均以稳定批大小尺度，该归约差异由验证集选择w₂但不能声称完全等价。m<2时对比项为可微零；角度目标仍可使用整个批次。

训练目标：joint=`L_angle+w₂L_cl`；cl_only=`w₂L_cl`；angle_only=`L_angle`。两个消融仅关闭对应组件，其余数据、初始化、学习率、种子、阈值、轮次和选模规则一致。CL-only是同条件组件基线，不冒充原论文完整SimCSE；angle-only不是SBERT。

`src/aligned_core.py`负责核心目标及历史函数加载；`src/train_aligned.py`负责训练和独立测试；`src/run_aligned.py`记录搜索计划、锁定配置、调度最终实验；`src/verify_aligned.py`独立检查式(4)与标量logsumexp实现的数值和梯度、关闭组件、空正例批次、历史角度函数。测试全部通过，证据为provenance/alignment/verification.json。

## 4 数据、训练与选模协议
STS-B来源mteb/stsbenchmark-sts，revision `{assets['dataset_revision']}`，训练5749、验证1500、测试1379条。基础模型revision `{assets['model_revision']}`，权重与官方LFS SHA256一致。保留原始划分；无序精确句对重叠为训练/验证7、训练/测试12、验证/测试3，另附去训练重叠的敏感性结果。

RTX5070 Ti Laptop 12GB、WSL2 Ubuntu24.04、PyTorch2.7.1+cu128。BERT-base uncased、CLS、max_length128、batch32、5epochs、AdamW lr5e-5、weight_decay0、warmup0、线性衰减、梯度裁剪1、BF16编码器与FP32损失。lr/CLS/τ有正文依据；batch/epoch/maxlen参考历史示例；精度和训练基础设施按现有硬件配置，不冒称原论文完全一致。

### 4.1 验证集搜索与锁定
{search_table}

选定w₂={selection['cl_weight']}、正例阈值t={selection['threshold']}，验证最高分{selection['validation_spearman']*100:.3f}。选择依据仅validation Spearman，并列按预声明顺序。配置和最终运行名单已写入selection.json，时间为{selection['selection_time']}。搜索脚本仅加载train/validation，搜索阶段不运行测试。最终五种子使用新种子142–146，全部模型训练完成后才统一评价test。旧版任务曾观察同一测试集，因此本轮是重新复现而非从未接触过测试数据的盲测；不能宣称更强的独立性。

## 5 实验结果
{main_table}

时间是训练、验证及检查点保存，不含依赖安装、权重下载和最终测试。误差为五种子样本标准差，不是标准误。

![主结果](../results_aligned/comparison.png)

![验证曲线](../results_aligned/validation_curves.png)

{seed_table}

原论文Table2列出AoE86.28±0.19、SBERT84.67±0.35、SimCSE76.27±0.23；这些是引用值，后两项本项目没有重跑。当前joint与论文AoE参考值相差{current-86.28:+.2f}个百分点，分数接近或不同都不能单独证明实现一致或不一致。

### 5.1 组件效应与统计边界
joint相对cl_only的平均差为{delta:+.3f}个百分点；对固定五组模型的测试样本进行1000次配对bootstrap，95%条件区间[{ci[0]:.3f}, {ci[1]:.3f}]。{conclusion}该区间不包含重新训练的不确定性；不把它等同于训练种子总体检验。

joint相对angle_only差值{bootstrap['angle_only']['mean_difference_points']:+.3f}，对应区间{bootstrap['angle_only']['conditional_test_item_95ci']}。配置在joint的验证结果上选取，消融使用同一配置，是受控组件分析，不是为每个方法独立调优后的最强性能比较。

去掉与训练重叠的测试项后剩余{int(mask.sum())}条，详细各次结果见results_aligned/nonoverlap_sensitivity.csv；这只排除精确句对重叠，不能排除语义近重复。

## 6 差异原因、限制与错误分析
可以确认的变化是移除额外cosine项、历史角度实现可追溯、对比目标按Eq.4独立核验、验证搜索及测试隔离。尚不能确认原文Table2的正例规则、权重最优值、每一步训练细节与初始化检查点。对比损失平均归约及BF16也属于明确差异，不能将其隐去。不能把未检验原因写成因果结论。

以下列出joint_seed142排序误差最大的四个真实句对；排序误差用人工分数排名与预测余弦排名的百分位差，不直接将不同量纲分数相减。全部20个案例在results_aligned/largest_rank_errors.csv。

'''
for _,r in first.sort_values('rank_error',ascending=False).head(4).iterrows():
    report+=f"- {r.id}：标注{r.score:.2f}，预测{r.prediction:.4f}，排序差{r.rank_error:.3f}；句1：{r.text1}；句2：{r.text2}\n"
report+='''
案例用于检查语义改写、否定或事件角色差异，不能据这四例推断整个测试集的类别分布；本轮未根据这些测试案例重新调参。

## 7 可复现命令与文件证据
所有命令从`/home/cdy/project/nlp`执行：

```bash
source .venv/bin/activate
python src/verify_aligned.py
python src/train_aligned.py --run-id smoke_check_v2 --smoke
python src/run_aligned.py
python src/report_aligned.py
python src/package_aligned.py
```

smoke只用64条训练/验证数据，不评价test。已有完成记录会复用，若要新跑smoke需换run-id。完整重跑应在项目副本中保留旧证据并另设空runs_aligned及空搜索计划/选择记录；不能把跳过已完成任务误称新训练。恢复环境与数据详见README，全部正式运行有config、history、train_result、metrics和逐条预测。基础模型与一个最终联合模型随包保存，其余模型可按固定配置重训；临时checkpoint不作为必要交付。

| 评分项 | 对应材料 | 仍需确认 |
|---|---|---|
| 复现范围 | §1、原论文Table2、plan.json | 与教师实际登记一致 |
| 方法实现 | §2–3、历史提交、aligned_core.py、verification.json | 未公开参数明确披露 |
| 结果分析 | §5–6、15组日志与预测、验证搜索表 | 文献基线不冒称本组重跑 |
| 可复现性 | 命令、环境锁、源代码、数据和哈希 | 新机器软件安装依赖 |
| 报告规范 | 引用、图表、真实结果与限制 | 提交格式符合课程要求 |
| 个人工作 | 末段contribution、本人复核记录和实验ID | 两人真实职责与证据待填 |
| 个人理解 | 能解释公式、正例阈值、验证选择和统计边界 | 不能用问答模板代替本人理解 |

## 8 结论与登记范围一致性
本轮已在硬件允许范围内完成双目标结构对齐、历史代码审计、6组验证搜索和15组五种子正式复现/消融。相比旧版，方法实现与论文核心目标的对应关系更明确；对原文未公开的正例处理和训练配置仍保持有限声明。个人贡献和教师范围确认不由实验脚本自动完成。

## 参考文献与来源
[1] Li, Xianming and Li, Jing. 2024. AoE: Angle-optimized Embeddings for Semantic Textual Similarity. ACL 2024, 1825–1839. https://aclanthology.org/2024.acl-long.101/

'''
report+=f'[2] 作者历史代码（2024-07-31，固定提交）：https://github.com/SeanLee97/AnglE/tree/{commit}\n\n'
report+='''[3] STS-B公开数据：https://huggingface.co/datasets/mteb/stsbenchmark-sts

[4] CCF人工智能目录：https://www.ccf.org.cn/Academic_Evaluation/AI/

## 组内贡献（最后一段，提交前据实填写）
成员甲：[姓名/学号待填写]，实际职责：[待填写]，具体产出与证据：[方法核对记录、代码路径或本人实验ID待填写]；成员乙：[姓名/学号待填写]，实际职责：[待填写]，具体产出与证据：[数据审计、统计复核、报告修改记录待填写]。本项目使用Codex辅助历史代码审计、实现、运行与报告生成；两人应据实确认实际完成工作，代码存在不自动证明个人作者身份，不补造commit或历史贡献。
'''
(ROOT/'report/report.md').write_text(report,encoding='utf8')
body=MarkdownIt('commonmark',{'html':False}).enable('table').render(report)
def embed(m):
    p=(ROOT/'report'/m.group(1)).resolve()
    assert p.is_relative_to(ROOT)
    return 'src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
body=re.sub(r'src="([^"]+)"',embed,body)
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>AoE方法对齐复现报告</title><style>body{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;max-width:1040px;margin:40px auto;padding:0 24px;line-height:1.8;color:#17212d}h1{font-size:28px}h2{font-size:22px;border-bottom:1px solid #ddd;padding-top:14px}h3{font-size:18px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ccd3dc;padding:8px;text-align:left}th{background:#eef3f8}img{max-width:100%}p,td{overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#f3f5f7;padding:12px}@media print{body{margin:0;font-size:10pt}h2,h3{break-after:avoid}img{break-inside:avoid}}</style>'''+body+'</html>'
(ROOT/'report/report.html').write_text(html,encoding='utf8')
print(summary.to_string());print('All 15 final runs verified; report rebuilt')
