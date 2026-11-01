import subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for _ in range(90):
    if len(list((root/'runs_aligned').glob('*/metrics.json')))==15:
        subprocess.run([sys.executable,str(root/'src/finalize_aligned.py')],check=True,cwd=root)
        break
    time.sleep(5)
else:
    raise RuntimeError('Final test results not yet complete; inspect pipeline log')
