# safety_bounds.py
import time
import torch
import numpy as np
from Transformer import TransformerGamma

# ---------- 模型 & γ 评分 ----------
device = torch.device("cpu")
model = TransformerGamma().to(device)
model.load_state_dict(torch.load("../models/network/transformer_gamma.pt", map_location=device))
model.eval()

@torch.no_grad()
def gamma_model(q_batch: torch.Tensor, dq_batch: torch.Tensor) -> torch.Tensor:
    x = torch.cat([q_batch, dq_batch], dim=1).to(device)
    _, gamma = model(x)
    return gamma.squeeze()


def online_search(q: torch.Tensor,
                                 qd: torch.Tensor,
                                 gamma_fn,
                                 a_max_np: np.ndarray,
                                 delta_t,
                                 tol,
                                 threshold):
    q = q.clone().float()
    qd = qd.clone().float()
    a_max = torch.tensor(a_max_np, dtype=torch.float32)

    # a = 0
    q_base = q + qd * delta_t  # shape [7]

    # 检查 q_base 是否安全
    # if gamma_fn(q_base.unsqueeze(0), qd.unsqueeze(0)).item() < threshold:
    #     raise RuntimeError("q_base already unsafe")

    q_min = torch.empty(7)
    q_max = torch.empty(7)

    # 对每个方向做并行二分
    for direction in ("pos", "neg"):
        deltaq = 0.5 * a_max * delta_t ** 2  # 最远可探距离
        if direction == "pos":
            lo = q_base.clone()
            hi = q_base + deltaq
        else:  # 负方向
            lo = q_base - deltaq
            hi = q_base.clone()

        # 保证 lo ≤ hi
        lo, hi = torch.min(lo, hi), torch.max(lo, hi)

        # 二分搜索直到区间宽度 ≤ tol
        while torch.any((hi - lo) > tol):
            mid = (lo + hi) / 2.0

            # 构造批量姿态：一次替换 7 个关节
            q_batch = q_base.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = mid
            qd_batch = qd.unsqueeze(0).repeat(7, 1)

            gamma_mid = gamma_fn(q_batch, qd_batch)  # shape [7]

            # 根据 γ 更新 lo / hi
            if direction == "pos":
                lo = torch.where(gamma_mid >= threshold, mid, lo)
                hi = torch.where(gamma_mid < threshold, mid, hi)
            else:  # neg
                hi = torch.where(gamma_mid >= threshold, mid, hi)
                lo = torch.where(gamma_mid < threshold, mid, lo)

        # 最终 lo/hi 已缩到 tol 内，取安全端作为边界
        if direction == "pos":
            q_max = lo.clone()  # lo 是最后一次判定安全的位置
        else:
            q_min = hi.clone()  # hi 是最后一次判定安全的位置

    # 容错检查
    # if torch.any(q_min > q_max):
    #     bad = (q_min > q_max).nonzero(as_tuple=True)[0].tolist()
    #     raise ValueError(f"q_min > q_max on joints {bad}")
    return q_min.numpy(), q_max.numpy()


def online_gridsearch(q: torch.Tensor,
                             qd: torch.Tensor,
                             gamma_fn,
                             a_max_np: np.ndarray,
                             delta_t,
                             step_size,
                             threshold):
    q     = q.clone().float()
    qd    = qd.clone().float()
    a_max = torch.tensor(a_max_np, dtype=torch.float32)

    # a=0,作为搜索起点
    q_base = q + qd * delta_t

    # 先用网络判定 q_base 是否安全
    q_batch = q_base.unsqueeze(0).repeat(7, 1)
    qd_batch = qd.unsqueeze(0).repeat(7, 1)
    gamma0 = gamma_fn(q_batch, qd_batch)          # shape [7]

    # 初始边界：若 q_base 已经碰撞，边界就是当前 q
    q_min = torch.where(gamma0 < threshold, q, q_base).clone()
    q_max = torch.where(gamma0 < threshold, q, q_base).clone()

    # 每个关节是否已经撞到（从 q_base 出发算）
    done = torch.zeros(7, dtype=torch.bool)                    # True ⇔ 已经不安全

    for direction, sign in (("pos", +1), ("neg", -1)):
        # 对于该方向，重新拷贝 done/current
        done_dir = done.clone()
        current  = q_base.clone()

        # 扫描 α = step, 2*step, …, 1
        alphas = torch.arange(step_size, 1.0 + 1e-6, step_size)
        for α in alphas:
            q_step = q_base + 0.5 * sign * α * a_max * delta_t**2

            # 只动一个关节批量前向
            q_batch = q_base.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = q_step
            qd_batch = qd.unsqueeze(0).repeat(7, 1)

            gamma = gamma_fn(q_batch, qd_batch)

            collided   = (~done_dir) & (gamma < threshold)
            still_safe = (~done_dir) & (gamma >= threshold)

            if collided.any():
                if direction == "pos":
                    q_max[collided] = current[collided]
                else:
                    q_min[collided] = current[collided]
                done_dir |= collided

            current[still_safe] = q_step[still_safe]

            if done_dir.all():
                break

        # 本方向始终安全的关节 → 边界放到最远一步
        unfinished = ~done_dir
        if unfinished.any():
            if direction == "pos":
                q_max[unfinished] = current[unfinished]
            else:
                q_min[unfinished] = current[unfinished]

    return q_min.numpy(), q_max.numpy()
__all__ = [
    "gamma_model",
    "online_search",
    "online_gridsearch",
]
