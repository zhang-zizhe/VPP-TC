import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from learning import TransformerGamma


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


def compute_safe_bounds_step(q, qd, gamma_fn, a_max_np,
                             delta_t=0.02, step_size=0.01,
                             threshold=2.5):
    """
    从 q 出发，沿正负方向按固定步长(例如0.1)扫描:
        q' = q + qd*dt + 0.5*sign*α*a_max*dt²,  α∈{step,2·step,…,1}
    若第一次 Γ<threshold，则回退一步作为边界。
    返回:
        q_min, q_max  (np.ndarray, shape [7])
    """
    q     = q.clone().float()
    qd    = qd.clone().float()
    a_max = torch.tensor(a_max_np, dtype=torch.float32)

    # 默认都落在当前位置 (万一一直 safe)
    q_min = q.clone()
    q_max = q.clone()

    for direction, sign in (("pos", +1), ("neg", -1)):
        done    = torch.zeros(7, dtype=torch.bool)   # 该关节是否已撞
        current = q.clone()                          # 上一次仍安全的位置

        # 依次尝试 α = 0.1, 0.2, …, 1.0
        alphas = torch.arange(step_size, 1.0 + 1e-6, step_size)
        for α in alphas:
            q_step = q + qd * delta_t + 0.5 * sign * α * a_max * delta_t**2

            # 每个关节分别替换成 q_step，批量前向 (一次 7 关节)
            q_batch = q.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = q_step
            qd_batch = qd.unsqueeze(0).repeat(7, 1)

            gamma = gamma_fn(q_batch, qd_batch)      # shape [7]

            # 对“还没撞”的关节检查是否越界
            collided   = (~done) & (gamma < threshold)
            still_safe = (~done) & (gamma >= threshold)

            # 记录刚发生碰撞的 -> 边界是 current
            if collided.any():
                if direction == "pos":
                    q_max[collided] = current[collided]
                else:
                    q_min[collided] = current[collided]
                done |= collided                      # 标记完成

            # 仍然安全的关节，更新 current，继续下一步
            current[still_safe] = q_step[still_safe]

            # 如果所有关节都已撞或走到 α=1 就可以停
            if done.all():
                break

        # 没撞到的关节 ⇒ 整段都 safe；边界 = 最远一步
        unfinished = ~done
        if unfinished.any():
            if direction == "pos":
                q_max[unfinished] = current[unfinished]
            else:
                q_min[unfinished] = current[unfinished]

    return q_min.numpy(), q_max.numpy()



if __name__ == "__main__":
    df = pd.read_csv("generated_q_qd_samples.csv")
    q_all  = torch.tensor(df.iloc[:, 0:7 ].values, dtype=torch.float32)
    qd_all = torch.tensor(df.iloc[:, 7:14].values, dtype=torch.float32)
    a_max  = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)

    total_time = 0.0
    all_bounds = []

    for q, qd in zip(q_all, qd_all):
        t0 = time.time()
        qmin, qmax = compute_safe_bounds_step(q, qd, gamma_model, a_max)
        total_time += time.time() - t0
        # 交错排列 q1_min,q1_max,q2_min,q2_max,...
        all_bounds.append(np.ravel(np.column_stack((qmin, qmax))))

    avg_time = total_time / len(q_all)
    print(f"Average time per sample: {avg_time:.4f} s")

    # 列名：q1_min,q1_max,...q7_max
    columns = [f"q{i+1}_{s}" for i in range(7) for s in ("min", "max")]
    pd.DataFrame(all_bounds, columns=columns).to_csv("estimated_bounds.csv", index=False)
    print("Results saved to estimated_bounds.csv")
