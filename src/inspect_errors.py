from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
p=pd.read_csv(ROOT/'runs/aoe_seed42/test_predictions.csv')
p['rank_error']=abs(rankdata(p.score)/len(p)-rankdata(p.prediction)/len(p))
print(p.sort_values('rank_error',ascending=False).head(8).to_json(orient='records',force_ascii=False,indent=2))
