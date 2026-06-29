"""Sobolev training: a distance regressor supervised on BOTH the value
(viability margin, metres) AND its gradient d(margin)/d[q,qd] (28-dim).

The barrier uses the GRADIENT, and a plain classifier/regressor leaves the
gradient UNsupervised -> unfaithful -> the controller flies blind and slips.
Matching the gradient directly fixes that root cause.

  loss = value_huber  +  lambda * grad_loss
  grad of the model is obtained by autograd through the input (double backprop).
"""
import os, sys, argparse, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from contextlib import nullcontext
try:                       # force MATH attention (efficient kernels lack 2nd-order grad)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _MATH = lambda: sdpa_kernel(SDPBackend.MATH)
except Exception:
    _MATH = nullcontext

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from vpptc.model import GammaRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="gradient-loss weight")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--coll-weight", type=float, default=8.0,
                    help="upweight value+grad loss on collision samples (md<0)")
    ap.add_argument("--val-weight", type=float, default=10.0,
                    help="weight on the value(distance) loss so RMSE actually converges")
    ap.add_argument("--save-every", type=int, default=10,
                    help="also dump an intermediate checkpoint every N epochs (0=off)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev}")

    raw = np.loadtxt(args.data, delimiter=",", skiprows=1, dtype=np.float32)
    X = torch.tensor(raw[:, 0:28])
    y = torch.tensor(raw[:, 28:29])           # min_dist (m)
    G = torch.tensor(raw[:, 29:57])            # gradient [gq14, gqd14]
    print(f"data {X.shape[0]} rows; unsafe {(y<0).float().mean()*100:.1f}%")
    ds = TensorDataset(X, y, G)
    g = torch.Generator().manual_seed(0)
    ntr = int(0.95*len(ds))
    tr, te = random_split(ds, [ntr, len(ds)-ntr], generator=g)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    el = DataLoader(te, batch_size=4096, num_workers=2)

    model = GammaRegressor(input_dim=28, d_model=args.d_model, nhead=args.nhead,
                           num_layers=args.num_layers, output_mode="linear").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    huber = nn.SmoothL1Loss(reduction="none", beta=0.005)

    def step(Xb, yb, Gb, train):
        Xb = Xb.to(dev).requires_grad_(True); yb = yb.to(dev); Gb = Gb.to(dev)
        with _MATH():
            out = model(Xb)
        if out.dim() == 2 and out.shape[1] == 1: out = out
        else: out = out.view(-1, 1)
        # input gradient via autograd (create_graph for double backprop in train)
        gpred = torch.autograd.grad(out.sum(), Xb, create_graph=train)[0]  # (B,28)
        w = torch.where(yb < 0, torch.full_like(yb, args.coll_weight), torch.ones_like(yb))
        v_loss = (w * huber(out, yb)).mean()
        # gradient loss = magnitude (MSE) + DIRECTION (1 - cosine).  The barrier
        # needs the direction right, so weight cosine heavily.
        gmse = (gpred - Gb).pow(2).mean(1, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(gpred, Gb, dim=1, eps=1e-8).unsqueeze(1)
        # DIRECTION-dominated: cosine is what the barrier actually needs.  gmse is
        # kept as a small magnitude pin (cosine is scale-invariant, so without it
        # the net could shrink the gradient to ~0 and still score high cosine).
        g_loss = (w * ((1.0 - cos) + 0.1 * gmse)).mean()
        return v_loss, g_loss

    out = args.output
    if not os.path.isabs(out): out = os.path.join(_ROOT, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    def save_ckpt(path):
        torch.save({"state_dict": model.state_dict(),
                    "config": dict(input_dim=28, d_model=args.d_model, nhead=args.nhead,
                                   num_layers=args.num_layers, output_mode="linear",
                                   max_dist=0.2)}, path)

    for ep in range(1, args.epochs+1):
        model.train(); vs=gs=0; nb=0
        for Xb, yb, Gb in tl:
            opt.zero_grad()
            v, gl = step(Xb, yb, Gb, True)
            (args.val_weight*v + args.lam*gl).backward()
            opt.step()
            vs+=v.item(); gs+=gl.item(); nb+=1
        sched.step()
        # eval: value RMSE + gradient cosine
        model.eval(); se=0.0; nse=0; se_b=0.0; nse_b=0; cos=[]
        tp=fp=n_uns=n_saf=0
        for Xb, yb, Gb in el:
            Xb2 = Xb.to(dev).requires_grad_(True); ybd = yb.to(dev)
            with _MATH():
                o = model(Xb2).view(-1,1)
            gp = torch.autograd.grad(o.sum(), Xb2)[0]
            cos.append(torch.nn.functional.cosine_similarity(gp, Gb.to(dev), dim=1).mean().item())
            err2 = (o-ybd)**2
            se += err2.sum().item(); nse += o.numel()
            near = (ybd.abs() < 0.02)                   # near-boundary = where the barrier acts
            se_b += err2[near].sum().item(); nse_b += int(near.sum().item())
            uns = (ybd < 0); pred_uns = (o < 0)         # derived classification
            n_uns += uns.sum().item(); n_saf += (~uns).sum().item()
            tp += (uns & pred_uns).sum().item(); fp += ((~uns) & pred_uns).sum().item()
        rmse = math.sqrt(se/nse)*1000
        rmse_b = math.sqrt(se_b/max(nse_b,1))*1000
        recall = tp/max(n_uns,1)*100; fpr = fp/max(n_saf,1)*100
        print(f"[{ep:3d}/{args.epochs}] v={vs/nb:.4f} g={gs/nb:.4f} | "
              f"RMSE={rmse:.1f}mm RMSE@bnd={rmse_b:.1f}mm  grad_cos={np.mean(cos):+.3f}  "
              f"recall@0={recall:.1f}% FP@0={fpr:.1f}%", flush=True)
        if args.save_every and ep % args.save_every == 0 and ep < args.epochs:
            ck = out.replace(".pt", f"_ep{ep}.pt"); save_ckpt(ck)
            print(f"  ckpt -> {ck}", flush=True)

    save_ckpt(out)
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
