# NLP 课程期中大作业：复现 AoE 句向量模型

本项目是自然语言处理（NLP）课程的期中大作业，复现 Li 和 Li 发表于 ACL 2024 主会的论文 [AoE: Angle-optimized Embeddings for Semantic Textual Similarity](https://aclanthology.org/2024.acl-long.101/)。我们选取论文表 2 中 **BERT-base 在 STS-B 上的域内实验**，研究一个具体问题：同时使用角度排序损失和对比损失，能否让模型对句子相似程度的排序更接近人工评分？

## 方法与复现范围

我们使用 BERT-base 编码句子，以最后一层的 CLS 向量作为句向量。角度排序损失使用 STS-B 的连续相似度评分，对比损失则从一个训练批次中选出高分句对作为正例。角度损失取自作者的历史代码，对比损失依据论文公式（4）实现。

原论文对表 2 实验的正例划分规则和部分训练配置说明有限。本项目在验证集上比较了 6 组配置，最终选定正例阈值为 4 分（满分 5 分），对比损失权重为 10。具体依据见[方法对照说明](provenance/alignment/METHOD_ALIGNMENT.md)。

## 实验结果

下表报告 STS-B 测试集上的 Spearman 等级相关系数乘以 100，数值越高，说明模型排序与人工评分越一致。本组结果为 5 个随机种子的均值 ± 样本标准差。

| 实验设置 | Spearman × 100 |
| --- | ---: |
| 原论文表 2：AoE | 86.28 ± 0.19 |
| 本组复现：角度排序损失 + 对比损失 | **85.58 ± 0.26** |
| 本组消融：仅角度排序损失 | 85.11 ± 0.41 |
| 本组消融：仅对比损失 | 77.51 ± 0.77 |

在本组配置下，联合训练比仅使用角度排序损失高约 0.47 分，比仅使用对比损失高约 8.07 分。联合训练的均值比原论文低 0.70 分。两组消融沿用联合训练选定的配置，未分别调优。完整统计见[实验结果摘要](results_aligned/summary.csv)。

## 运行方法

实测环境为 WSL2、Ubuntu 24.04、Python 3.12 和 NVIDIA RTX 5070 Ti Laptop（12 GB 显存）。在仓库根目录依次执行以下命令。首次准备数据时需要联网下载 BERT 权重和 STS-B 数据集。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r provenance/requirements-lock.txt
python src/prepare.py
```

先用小样例检查训练流程：

```bash
python src/train_aligned.py --run-id smoke_check --smoke
```

再运行完整实验，包括验证集参数搜索、三个实验设置各五次训练，以及最终测试：

```bash
python src/run_aligned.py
```

运行日志与逐次实验结果保存在 `runs_aligned/`。核心损失函数见 [aligned_core.py](src/aligned_core.py)，训练及评价流程见 [train_aligned.py](src/train_aligned.py)。本次作业使用的模型、数据版本及校验值见[资源记录](provenance/assets.json)。