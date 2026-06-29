#!/usr/bin/env python3
"""margin calibration training: add MSE on the classifier margin (gamma=logit0-logit1) to pin it to the clean distance.
loss = CE(logits,y) + lam * MSE(gamma, SCALE*clean_dist)
=> gamma becomes a calibrated (signed) distance: at the same dist, gamma no longer floats/saturates freely.
The decision is still sign(gamma). Precondition: clean_dist must be clean (structural pairs already removed).
  python3 scripts/train/fast_train_margin.py <data.csv> <out.pt> <epochs> [lam] [scale]
"""
import sys, os, math, time
os.chdir('/home/yicong/openarm_test'); sys.path.insert(0, '/home/yicong/openarm_test')
import numpy as np, pandas as pd, torch, torch.nn as nn
from vpptc.transformer_gamma_orig import TransformerGamma
from sklearn.metrics import recall_score
DATA, OUT, EPOCHS = sys.argv[1], sys.argv[2], int(sys.argv[3])
LAM = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
SCALE = float(sys.argv[5]) if len(sys.argv) > 5 else 200.0   # gamma target = SCALE*dist(m); 200 => unit 5mm
PROG = 'scratch_vhacd/_train_progress.txt'
dev = 'cuda'; BS = 4096; LR = 2e-4; WARMUP = 3
print(f'loading {DATA}  (lam={LAM}, scale={SCALE})', flush=True)
df = pd.read_csv(DATA)
X = torch.tensor(df.iloc[:, 0:28].values, dtype=torch.float32)
dist = df['min_dist'].values.astype(np.float32)
y = torch.tensor((dist < 0).astype(np.int64))
MODE = sys.argv[6] if len(sys.argv) > 6 else 'linear'      # linear | tanh
D0 = float(sys.argv[7]) if len(sys.argv) > 7 else 0.015    # tanh knee point (m)
if MODE == 'tanh':
    tgt = torch.tensor(SCALE * np.tanh(dist / D0), dtype=torch.float32)   # steep near 0, saturates far away, consistent
else:
    tgt = torch.tensor(dist * SCALE, dtype=torch.float32)
distm = torch.tensor(dist * 1000, dtype=torch.float32)     # true dist (mm), for eval bucketing
n = len(X); print(f'{n} samples, mode={MODE} D0={D0}, unsafe {(y==1).float().mean()*100:.1f}%, dist[{dist.min()*1000:.0f},{dist.max()*1000:.0f}]mm', flush=True)
g = torch.Generator().manual_seed(42); perm = torch.randperm(n, generator=g); ntr = int(0.8 * n)
Xtr, ytr, ttr = X[perm[:ntr]].to(dev), y[perm[:ntr]].to(dev), tgt[perm[:ntr]].to(dev)
Xte, yte, tte = X[perm[ntr:]].to(dev), y[perm[ntr:]].to(dev), tgt[perm[ntr:]].to(dev)
dte = distm[perm[ntr:]].to(dev)  # test dist in mm
model = TransformerGamma(input_dim=28, d_model=128, nhead=4, num_layers=4, dropout=0.1, dim_feedforward=512).to(dev)
cmodel = torch.compile(model)
ce = nn.CrossEntropyLoss(); opt = torch.optim.AdamW(model.parameters(), lr=LR); scaler = torch.amp.GradScaler('cuda')
spe = (ntr + BS - 1) // BS; warm = WARMUP * spe; total = EPOCHS * spe
def lr_lambda(s):
    if s < warm: return s / max(1, warm)
    pr = (s - warm) / max(1, total - warm); return 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
def save(path):
    torch.save({'state_dict': model.state_dict(),
                'config': dict(input_dim=28, d_model=128, nhead=4, num_layers=4, dropout=0.1, dim_feedforward=512),
                'model_type': 'TransformerGamma', 'margin_scale': SCALE}, path)
def evalcalib():
    model.eval()
    with torch.no_grad():
        gl = []; gm = []
        for i in range(0, len(Xte), 16384):
            lg, ga = model(Xte[i:i+16384]); gl.append(lg.argmax(1)); gm.append(ga)
        pr = torch.cat(gl); gam = torch.cat(gm)
    acc = (pr == yte).float().mean().item()
    rec = recall_score(yte.cpu().numpy(), pr.cpu().numpy(), pos_label=1, zero_division=0)
    # gamma variance at the same dist: take a narrow bucket of dist in [0,2]mm
    d = dte; gam_c = gam
    m = (d > 0) & (d < 2)
    std_band = gam_c[m].std().item() if m.sum() > 50 else float('nan')
    corr = torch.corrcoef(torch.stack([gam_c, d]))[0, 1].item()
    return acc, rec, std_band, corr
t0 = time.perf_counter()
for ep in range(1, EPOCHS + 1):
    model.train(); idx = torch.randperm(ntr, device=dev); L=torch.zeros((),device=dev); Lc=torch.zeros((),device=dev); Lr=torch.zeros((),device=dev)
    for i in range(0, ntr, BS):
        b = idx[i:i+BS]; xb, yb, tb = Xtr[b], ytr[b], ttr[b]; opt.zero_grad()
        with torch.amp.autocast('cuda'):
            lg, gam = cmodel(xb)
            lc = ce(lg, yb); lr = ((gam - tb) ** 2).mean(); loss = lc + LAM * lr
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        L += loss.detach()*len(b); Lc += lc.detach()*len(b); Lr += lr.detach()*len(b)
    acc, rec, std_band, corr = evalcalib()
    el = time.perf_counter() - t0
    line = (f'[Ep {ep:3d}/{EPOCHS}] loss={L.item()/ntr:.3f}(ce={Lc.item()/ntr:.3f} reg={Lr.item()/ntr:.2f}) '
            f'acc={acc*100:.1f}% rec={rec*100:.1f}% | gamma@0-2mm std={std_band:.2f} corr(g,dist)={corr:.3f}  elapsed{el/60:.1f}m ETA{el/ep*(EPOCHS-ep)/60:.1f}m')
    print(line, flush=True); open(PROG, 'w').write(line + '\n')
    if ep % 25 == 0 and ep < EPOCHS: save(OUT.replace('.pt', f'_ep{ep}.pt'))
save(OUT); print('SAVED ' + OUT, flush=True); open(PROG, 'w').write('DONE ' + OUT + '\n')
