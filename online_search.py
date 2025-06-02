import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from learning import TransformerGamma

# ─────────────────────────────────────────────────────────────
# 0. 设备 & 模型加载（CPU 强制，以免 NCCL 报错）
# ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
model = TransformerGamma().to(device)
model.load_state_dict(torch.load("../../PythonProject3/torque-constraint-learning/transformer_gamma.pt", map_location=device))
model.eval()

@torch.no_grad()
def gamma_model(q_batch, dq_batch):
    """批量计算 Γ 分数，返回 shape [N]"""
    x = torch.cat([q_batch, dq_batch], dim=1).to(device)
    _, gamma = model(x)
    return gamma.squeeze()          # shape: [N]

# ─────────────────────────────────────────────────────────────
# 1. 并行二分搜索函数（一次前向即算 7 个关节）
# ─────────────────────────────────────────────────────────────
def compute_safe_bounds_parallel(q, qd, gamma_fn, a_max_np,
                                 delta_t=0.02, tol=1e-3, threshold=2.5):
    """
    输入:
        q, qd      : torch.Tensor shape [7]
        a_max_np   : np.ndarray shape [7]
    输出:
        q_min, q_max : np.ndarray shape [7]
    """
    a_max = torch.tensor(a_max_np, dtype=torch.float32)          # 保证 dtype 一致
    q = q.clone()
    qd = qd.clone()

    q_min = torch.empty(7)
    q_max = torch.empty(7)

    for direction in ("pos", "neg"):
        # ① 初始化搜索区间
        if direction == "pos":
            hi = q + qd * delta_t + 0.5 * a_max * delta_t**2
            lo = q.clone()
        else:
            lo = q + qd * delta_t - 0.5 * a_max * delta_t**2
            hi = q.clone()

        lo, hi = torch.min(lo, hi), torch.max(lo, hi)

        # ② 二分搜索
        while torch.any((hi - lo) > tol):
            mid = ((lo + hi) / 2.0).float()           # 确保 float32
            # 构造批量姿态，替换单个关节为 mid
            q_batch = q.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = mid
            qd_batch = qd.unsqueeze(0).repeat(7, 1)

            gamma_vals = gamma_fn(q_batch, qd_batch)  # shape [7]

            if direction == "pos":
                lo = torch.where(gamma_vals >= threshold, mid, lo)
                hi = torch.where(gamma_vals <  threshold, mid, hi)
            else:
                hi = torch.where(gamma_vals >= threshold, mid, hi)
                lo = torch.where(gamma_vals <  threshold, mid, lo)

        if direction == "pos":
            q_max = lo.clone()
        else:
            q_min = hi.clone()

    return q_min.numpy(), q_max.numpy()

# ─────────────────────────────────────────────────────────────
# 2. 主流程：读取样本 → 计算 → 保存
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("generated_q_qd_samples.csv")
    q_all  = torch.tensor(df.iloc[:, 0:7 ].values, dtype=torch.float32)
    qd_all = torch.tensor(df.iloc[:, 7:14].values, dtype=torch.float32)
    a_max  = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)

    total_time = 0.0
    all_bounds = []

    for q, qd in zip(q_all, qd_all):
        t0 = time.time()
        qmin, qmax = compute_safe_bounds_parallel(q, qd, gamma_model, a_max)
        total_time += time.time() - t0
        # 交错排列 q1_min,q1_max,q2_min,q2_max,...
        all_bounds.append(np.ravel(np.column_stack((qmin, qmax))))

    avg_time = total_time / len(q_all)
    print(f"📊 Average time per sample: {avg_time:.4f} s")

    # 列名：q1_min,q1_max,...q7_max
    columns = [f"q{i+1}_{s}" for i in range(7) for s in ("min", "max")]
    pd.DataFrame(all_bounds, columns=columns).to_csv("estimated_bounds.csv", index=False)
    print("✅ Results saved to estimated_bounds.csv")
