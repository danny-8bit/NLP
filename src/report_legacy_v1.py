"""Build tables, plots, and a report from completed experiments only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results';out.mkdir(exist_ok=True)
paths=[p for p in (ROOT/'runs').glob('*/metrics.json') if '_smoke' not in str(p)]
rows=[json.loads(p.read_text()) for p in paths]
expected={(m,s) for m in ['aoe','no_angle'] for s in range(42,47)}|{('untrained',42),('cosent',42)}
actual={(r['method'],r['seed']) for r in rows}
missing=expected-actual
if missing:raise RuntimeError(f'Cannot generate final report. Missing runs: {sorted(missing)}')
df=pd.DataFrame(rows).sort_values(['method','seed']);df.to_csv(out/'all_runs.csv',index=False)
summary=df.groupby('method').agg(n=('seed','count'),spearman_mean=('spearman','mean'),
    spearman_std=('spearman','std'),pearson_mean=('pearson','mean'),seconds_mean=('seconds','mean'),
    peak_memory_gib=('peak_memory_gib','max'))
summary.to_csv(out/'summary.csv')
training=[json.loads(line) for line in (ROOT/'data/train.jsonl').read_text().splitlines()]
train_pairs={tuple(sorted((r['text1'],r['text2']))) for r in training}
clean_rows=[]
for r in rows:
    p=pd.read_csv(ROOT/f"runs/{r['method']}_seed{r['seed']}/test_predictions.csv")
    mask=[tuple(sorted((str(a),str(b)))) not in train_pairs for a,b in zip(p.text1,p.text2)]
    p=p[mask]
    clean_rows.append({'method':r['method'],'seed':r['seed'],'n':len(p),'spearman':float(spearmanr(p.score,p.prediction).statistic)})
clean=pd.DataFrame(clean_rows)
clean.to_csv(out/'nonoverlap_sensitivity.csv',index=False)
clean_summary=clean.groupby('method').spearman.agg(['mean','std'])
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(summary.index,100*summary.spearman_mean,yerr=100*summary.spearman_std.fillna(0),capsize=5)
ax.set_ylabel('Test Spearman x 100');ax.set_title('STS-B: mean and seed standard deviation');fig.tight_layout()
fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for method in ['aoe','no_angle','cosent']:
    h=json.loads((ROOT/f'runs/{method}_seed42/history.json').read_text())
    ax.plot([r['epoch'] for r in h],[r['spearman']*100 for r in h],marker='o',label=method)
ax.legend();ax.set_xlabel('Epoch');ax.set_ylabel('Validation Spearman x 100');fig.tight_layout()
fig.savefig(out/'validation_curves.png',dpi=180);plt.close(fig)
pred=pd.read_csv(ROOT/'runs/aoe_seed42/test_predictions.csv')
pred['rank_error']=np.abs(rankdata(pred.score)/len(pred)-rankdata(pred.prediction)/len(pred))
pred.sort_values('rank_error',ascending=False).head(20).to_csv(out/'largest_rank_errors.csv',index=False)
# Bootstrap over test items of mean per-seed Spearman differences; not an independent
# test of population seed variability, and no hyperparameter selection on these draws.
aoe=np.array([pd.read_csv(ROOT/f'runs/aoe_seed{s}/test_predictions.csv').prediction for s in range(42,47)])
base=np.array([pd.read_csv(ROOT/f'runs/no_angle_seed{s}/test_predictions.csv').prediction for s in range(42,47)])
y=pred.score.to_numpy();rng=np.random.default_rng(2026);deltas=[]
for _ in range(1000):
    ids=rng.integers(0,len(y),len(y))
    deltas.append(np.mean([spearmanr(y[ids],a[ids]).statistic-spearmanr(y[ids],b[ids]).statistic for a,b in zip(aoe,base)]))
ci=np.quantile(deltas,[.025,.975])*100
delta=100*(summary.loc['aoe','spearman_mean']-summary.loc['no_angle','spearman_mean'])
(out/'bootstrap.json').write_text(json.dumps({'difference_points':delta,'conditional_test_item_95ci':ci.tolist(),'draws':1000},indent=2))
table=['| 方法 | 次数 | Spearman×100 | Pearson×100 | 平均耗时/秒 | 峰值显存/GiB |','|---|---:|---:|---:|---:|---:|']
for m,r in summary.iterrows():
    score=f'{r.spearman_mean*100:.2f}'
    if r.n>1:score+=f' ± {r.spearman_std*100:.2f}'
    table.append(f'| {m} | {int(r.n)} | {score} | {r.pearson_mean*100:.2f} | {r.seconds_mean:.1f} | {r.peak_memory_gib:.2f} |')
paper_gap=summary.loc['aoe','spearman_mean']*100-86.28
assets=json.loads((ROOT/'provenance/assets.json').read_text())
verification=json.loads((ROOT/'provenance/verification.json').read_text())
best_epoch=df[df.method=='aoe'].selected_epoch.tolist()
direction='高于' if delta>0 else '低于或等于'
report=f'''# AoE语义文本相似度方法的部分复现与角度损失消融

课程期中实验报告；成员甲：[姓名/学号待填写]；成员乙：[姓名/学号待填写]。

## 摘要
本文复现ACL2024主会论文AoE的BERT-base/STS-B域内实验，并以移除角度损失开展受控消融。全部数值由本机实际实验生成，不复制论文分数作为自身结果。AoE五次测试Spearman均值为{summary.loc['aoe','spearman_mean']*100:.2f}，相对无角度对照变化{delta:+.2f}个百分点。当前实现为公开官方损失代码下的受控部分复现，存在训练细节和实现版本差异。

## 1 论文与任务
Xianming Li and Jing Li. 2024. AoE: Angle-optimized Embeddings for Semantic Textual Similarity. Proceedings of ACL 2024 (Volume 1: Long Papers), pp.1825–1839. DOI:10.18653/v1/2024.acl-long.101。ACL为CCF人工智能A类会议。正式论文见provenance/paper.pdf。本文聚焦原文§5.1/Table2的STS-B列，不包含GIS、NLI大规模训练、LLaMA与MTEB全套实验。

STS任务输入句对并预测语义相似度。BERT最后层CLS向量作为句子表示；推理取两向量余弦值。主指标为Spearman排序相关系数，附报Pearson。无需使用测试集拟合分数校准。

## 2 方法原理与实现
AoE将句向量拆为实部与虚部，在复数空间中构造角度差异，使用标签排序约束优化相似句对的表示，并结合辅助对比目标。公式以原论文Eq.2–5及归档官方loss.py为准；本项目直接调用官方angle_loss、cosine_loss、in_batch_negative_loss，避免手写公式与公开实现不一致。温度代码中的tau=20是乘法尺度，对应常见倒温度0.05，不能误写成tau=0.05再传给代码。

本项目AoE目标为cosine排序损失+二值批内对比损失+0.02×角度损失。无角度组只移除最后一项；cosent组仅保留cosine排序损失，不能冒充原文SBERT/SimCSE。未训练组为BERT CLS参考线。

## 3 数据与实验设计
数据为公开mteb/stsbenchmark-sts，revision={assets['dataset_revision']}；各划分数量为{assets['sizes']}。训练只使用train，选模只使用validation。对原始数据划分进行句对审计，发现无序精确句对重叠为{verification['exact_unordered_pair_overlap']}。这属于公开数据内在重叠，不能声称零重叠。主结果保留标准划分，另对去除训练重叠后的测试样本做预先声明的敏感性分析，不据此重新选参。原始分数0–5用于排序；仅批内对比目标用score≥3构成二值正标签。阈值提前固定，未按测试集调参。原始Arrow快照、JSONL和SHA256均随项目保存。

训练初始化为google-bert/bert-base-uncased，revision={assets['model_revision']}。硬件为RTX5070 Ti Laptop GPU约12GB，WSL2 Ubuntu24.04。采用batch32、max_length128、epoch5、AdamW学习率5e-5、weight_decay0、warmup0、梯度裁剪1.0；BF16编码器/FP32损失。随机种子42–46。每轮验证，根据验证Spearman保存最优模型，之后仅评价一次测试集；AoE所选epoch分别为{best_epoch}。完整依赖与GPU信息见provenance。

## 4 实验结果
{chr(10).join(table)}

![主结果](../results/comparison.png)

![验证曲线](../results/validation_curves.png)

AoE均值{direction}无角度组，差值为{delta:+.2f}个Spearman百分点。按相同种子配对并对测试样本进行1000次bootstrap，平均相关系数差的条件95%区间为[{ci[0]:.2f}, {ci[1]:.2f}]。该区间仅反映固定已训练模型下测试样本重采样不确定性，不等同于训练种子总体显著性，也不据此保证方法在其他任务有效。误差条是种子样本标准差。

去掉与训练集重复的测试句对后，剩余{clean_rows[0]['n']}条；AoE五种子平均Spearman×100为{clean_summary.loc['aoe','mean']*100:.2f}，无角度组为{clean_summary.loc['no_angle','mean']*100:.2f}。这是补充敏感性分析，不替代标准STS-B分数。

## 5 与原文比较及复现限制
论文Table2 STS-B AoE为86.28±0.19，本项目均值差为{paper_gap:+.2f}个百分点。该差值不是复现成功率。原文对STS-B专项实验的batch、epoch、最优权重等细节在当前所读正文未完整列出；本项目按明确预设配置执行，不声称严格历史重现。当前官方版本额外包含cosine项，连续STS标签转换为对比标签的适配也属实现差异；BF16精度、验证选择方式、现代依赖和硬件均可能影响结果。因此只能支持本设置下的经验结论，不能把差距唯一归因于某个组件。

## 6 误差分析
已保存20个排序误差最大的句对于results/largest_rank_errors.csv。使用预测排序百分位与人工排序百分位差，避免将原始余弦值与0–5标注直接相减产生尺度伪误差。以下列出四个代表案例；其具体错误类型需结合语义判断，不能仅凭关键词推断。

'''
for _,r in pred.sort_values('rank_error',ascending=False).head(4).iterrows():
    report+=f'- {r.id}：标注{r.score:.2f}，预测余弦{r.prediction:.4f}，排序差{r.rank_error:.3f}。句1：{r.text1}；句2：{r.text2}\n'
report+='''
对上述句对的语义分析如下：test-745的taste与preference在此都表达个人偏好，但模型对非词面一致的同义改写评分偏低；test-870的“试验并找到喜欢的”与“找到适合自己的”被数据标注为同义，模型可能过于关注局部措辞。test-804中两个句子共享rule/against等词语，但存在有无规则的否定对立，模型仍给出较高相似度，说明词面重合不能替代否定语义理解。test-721还包含额外的“You don't”，其语用和语境依赖可能加大判断难度。以上是Codex辅助对实际预测案例作出的解释性观察，需组员复核，不是因果证明或整个数据集的类别统计，也未据此修改当前测试模型。

## 7 复现操作与期末延伸
README列出了环境安装、数据准备、smoke检查、正式实验、报告生成及打包指令。代码与数据可离线复查，基础模型也随提交包保存。期末可在固定基线下尝试角度权重调度或池化改进，先在验证集选参，再用独立固定测试评价；不把此次消融自动包装成创新提升。

## 8 结论
本项目完成BERT-base在STS-B上的实际训练、验证选模、五种子角度损失消融、未训练及cosine排序参考线、逐条预测与资源统计。结果只适用于所列设置。完整论文实验、严格历史配置匹配以及创新有效性仍超出当前范围。教师若要求全部原表，需补充实验。

## 参考资料
- 正式论文：https://aclanthology.org/2024.acl-long.101/
- 官方代码：https://github.com/SeanLee97/AnglE
- 数据：https://huggingface.co/datasets/mteb/stsbenchmark-sts
- CCF目录：https://www.ccf.org.cn/Academic_Evaluation/AI/

## 组内贡献（提交前据实确认）
成员甲：[姓名/学号待填写]，拟负责论文阅读、环境部署、官方实现核对与训练实验；成员乙：[姓名/学号待填写]，拟负责数据核验、评价统计、误差分析、报告与打包。以上为待确认的分工模板，须按实际完成情况修改。实验搭建、代码与报告生成使用Codex辅助，组员须自行复核结果及课程对工具使用的要求。
'''
from report_rubric import enhance
report=enhance(report,ROOT)
(ROOT/'report/report.md').write_text(report,encoding='utf8')
from markdown_it import MarkdownIt
import base64,re
body=MarkdownIt('commonmark',{'html':False}).enable('table').render(report)
def embed_image(match):
    path=(ROOT/'report'/match.group(1)).resolve()
    if path.suffix=='.png' and path.is_relative_to(ROOT):
        return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
    return match.group(0)
body=re.sub(r'src="([^"]+)"',embed_image,body)
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>AoE期中实验报告</title>
<style>body{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;max-width:940px;margin:40px auto;padding:0 24px;color:#17212d;line-height:1.85}h1{font-size:28px}h2{font-size:22px;border-bottom:1px solid #ddd;padding-top:18px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd3dc;padding:8px;text-align:left}th{background:#eef3f8}img{max-width:100%}a{color:#185ca1}p{overflow-wrap:anywhere}@media print{body{margin:0;max-width:none;font-size:10pt}h2{break-after:avoid}img,table{break-inside:avoid}}</style>'''+body+'</html>'
(ROOT/'report/report.html').write_text(html,encoding='utf8')
print(summary.to_string());print('Report generated from',len(rows),'completed runs')
