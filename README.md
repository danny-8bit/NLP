# AoE on STS-B

Course reproduction of the BERT-base in-domain experiment in Table 2 of [Li & Li, ACL 2024](https://aclanthology.org/2024.acl-long.101/). The joint model achieved 85.58 ± 0.26 Spearman × 100 across five seeds. See results_aligned/summary.csv and provenance/alignment/METHOD_ALIGNMENT.md for the ablations and reconstruction assumptions.

From the project root on Ubuntu 24.04 with Python 3.12 and an NVIDIA GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r provenance/requirements-lock.txt
python src/prepare.py
python src/train_aligned.py --run-id smoke_check --smoke
python src/run_aligned.py
```

The preparation script downloads model weights and STS-B. The original asset revisions and checksums are recorded in provenance/assets.json. Third-party AnglE files are accompanied by their upstream licenses.
