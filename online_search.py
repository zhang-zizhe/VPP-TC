import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from learning import TransformerGamma


device = torch.device("cpu")
model = TransformerGamma().to(device)
model.load_state_dict(
    torch.load("../../PythonProject3/torque-constraint-learning/transformer_gamma.pt",
               map_location=device)
)
model.eval()


@torch.no_grad()
def gamma_model(q_batch: torch.Tensor, dq_batch: torch.Tensor) -> torch.Tensor:
    """批量计算 Γ 分数，返回 shape [N]"""
    x = torch.cat([q_batch, dq_batch], dim=1).to(device)
    _, gamma = model(x)
    return gamma.squeeze()          # shape: [N]


def compute_safe_bounds_parallel(q: torch.Tensor,
                                 qd: torch.Tensor,
                                 gamma_fn,
                                 a_max_np: np.ndarray,
                                 delta_t: float = 0.02,
                                 tol: float = 1e-3,
                                 threshold: float = 2.5):
    """
    输入:
        q, qd      : torch.Tensor, shape [7]
        a_max_np   : np.ndarray, shape [7]
    输出:
        q_min, q_max : np.ndarray, shape [7]
    """
    # 先检查当前位置是否安全
    gamma0 = gamma_fn(q.unsqueeze(0), qd.unsqueeze(0)).item()
    if gamma0 < threshold:
        raise RuntimeError(
            f"Initial (q, qd) already unsafe: Γ={gamma0:.3f} < {threshold}"
        )

    a_max = torch.tensor(a_max_np, dtype=torch.float32)
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

        # 二分搜索
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

    # 搜索结束后确认区间合法
    if torch.any(q_min > q_max):
        bad_idx = (q_min > q_max).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"q_min > q_max for joints {bad_idx} "
            f"(q_min={q_min.numpy()}, q_max={q_max.numpy()})"
        )

    return q_min.numpy(), q_max.numpy()



if __name__ == "__main__":
    df = pd.read_csv("generated_q_qd_samples.csv")
    q_all  = torch.tensor(df.iloc[:, 0:7 ].values, dtype=torch.float32)
    qd_all = torch.tensor(df.iloc[:, 7:14].values, dtype=torch.float32)
    a_max  = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)

    total_time = 0.0
    all_bounds = []

    for idx, (q, qd) in enumerate(zip(q_all, qd_all), 1):
        try:
            t0 = time.time()
            qmin, qmax = compute_safe_bounds_parallel(q, qd, gamma_model, a_max)
            total_time += time.time() - t0
            # 交错排列 q1_min,q1_max,q2_min,q2_max,...
            all_bounds.append(np.ravel(np.column_stack((qmin, qmax))))
        except (RuntimeError, ValueError) as e:
            print(f"❌ sample {idx} skipped – {e}")

    # 只保存成功样本
    if all_bounds:
        avg_time = total_time / len(all_bounds)
        print(f"Average time per VALID sample: {avg_time:.4f} s")

        columns = [f"q{i+1}_{s}" for i in range(7) for s in ("min", "max")]
        pd.DataFrame(all_bounds, columns=columns).to_csv("estimated_bounds.csv", index=False)
        print("Results saved to estimated_bounds.csv")
    else:
        print("No valid samples to save.")
