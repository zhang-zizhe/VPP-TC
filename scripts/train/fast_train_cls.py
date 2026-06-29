#!/usr/bin/env python3
"""Train d128 classifier with architecture+recipe identical to v012, with data resident on GPU for speedup (mathematically equivalent).
TransformerGamma d128/nhead4/L4/ffn512, AdamW lr2e-4, CrossEntropy, AMP,
cosine+warmup3, bs4096, label y=(min_dist<0), 80/20 split seed42.
  python3 scripts/train/fast_train_cls.py <data.csv> <out.pt> <epochs>
"""
import sys, os, math, time
os.chdir('/home/yicong/openarm_test'); sys.path.insert(0, '/home/yicong/openarm_test')
import numpy as np, pandas as pd, torch, torch.nn as nn
from vpptc.transformer_gamma_orig import TransformerGamma
from sklearn.metrics import recall_score
DATA, OUT, EPOCHS = sys.argv[1], sys.argv[2], int(sys.argv[3])
PROG = 'scratch_vhacd/_train_progress.txt'
dev = 'cuda'; BS = 4096; LR = 2e-4; WARMUP = 3
print('loading', DATA, flush=True)
df = pd.read_csv(DATA)
X = torch.tensor(df.iloc[:, 0:28].values, dtype=torch.float32)
y = torch.tensor((df['min_dist'].values < 0).astype(np.int64))
n = len(X); print(f'{n} samples, unsafe {(y==1).float().mean()*100:.1f}%', flush=True)
g = torch.Generator().manual_seed(42); perm = torch.randperm(n, generator=g); ntr = int(0.8 * n)
Xtr, ytr = X[perm[:ntr]].to(dev), y[perm[:ntr]].to(dev)
Xte, yte = X[perm[ntr:]].to(dev), y[perm[ntr:]].to(dev)
model = TransformerGamma(input_dim=28, d_model=128, nhead=4, num_layers=4, dropout=0.1, dim_feedforward=512).to(dev)
cmodel = torch.compile(model)  # JIT speedup, mathematically equivalent; use cmodel for training, use original model for saving/eval
crit = nn.CrossEntropyLoss(); opt = torch.optim.AdamW(model.parameters(), lr=LR); scaler = torch.amp.GradScaler('cuda')
spe = (ntr + BS - 1) // BS; warm = WARMUP * spe; total = EPOCHS * spe
def lr_lambda(s):
    if s < warm: return s / max(1, warm)
    pr = (s - warm) / max(1, total - warm); return 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
def save(path):
    torch.save({'state_dict': model.state_dict(),
                'config': dict(input_dim=28, d_model=128, nhead=4, num_layers=4, dropout=0.1, dim_feedforward=512),
                'model_type': 'TransformerGamma'}, path)
t0 = time.perf_counter()
for ep in range(1, EPOCHS + 1):
    model.train(); idx = torch.randperm(ntr, device=dev); tl = torch.zeros((), device=dev)
    for i in range(0, ntr, BS):
        b = idx[i:i + BS]; xb, yb = Xtr[b], ytr[b]; opt.zero_grad()
        with torch.amp.autocast('cuda'):
            lg, _ = cmodel(xb); loss = crit(lg, yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += loss.detach() * len(b)
    model.eval()
    with torch.no_grad():
        pr = torch.cat([model(Xte[i:i + 16384])[0].argmax(1) for i in range(0, len(Xte), 16384)])
    acc = (pr == yte).float().mean().item()
    rec = recall_score(yte.cpu().numpy(), pr.cpu().numpy(), pos_label=1, zero_division=0)
    el = time.perf_counter() - t0
    line = f'[Ep {ep:3d}/{EPOCHS}] loss={tl.item()/ntr:.4f} acc={acc*100:.2f}% recall={rec*100:.2f}%  elapsed {el/60:.1f}m ETA{el/ep*(EPOCHS-ep)/60:.1f}m'
    print(line, flush=True); open(PROG, 'w').write(line + '\n')
    if ep % 25 == 0 and ep < EPOCHS: save(OUT.replace('.pt', f'_ep{ep}.pt'))
save(OUT); print('SAVED', OUT, flush=True); open(PROG, 'w').write('DONE ' + OUT + '\n')
