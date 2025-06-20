import torch
import numpy as np
from Zonotope import ZonotopeNet

# 设备 & 模型加载（保持和你原来一致）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '../models/network/zonotope_net_1m2.pt'
# model_path = '../models/network/zonotope_net.pth'
model = ZonotopeNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def predict_zonotope(q, dq, gamma, acc_max, dt=0.02):
    """
    输入：
        q       -- array_like, shape=(7,), 当前关节位置
        dq      -- array_like, shape=(7,), 当前关节速度
        gamma   -- float, 或 shape=(1,)
        acc_max -- array_like, shape=(7,), 每个关节的最大加速度
        dt      -- float, 时间步长 (s)
    返回：
        qmin_clamped -- np.ndarray, shape=(7,), 截断后的下界
        qmax_clamped -- np.ndarray, shape=(7,), 截断后的上界
    如果某个维度出现 qmin_clamped[i] > qmax_clamped[i]，则只对该维度
    将 qmin_clamped[i]=min_pos[i], qmax_clamped[i]=max_pos[i]，并打印警告。
    """
    # 构造输入并前向推理
    x = np.concatenate([
        np.asarray(q, dtype=np.float32),
        np.asarray(dq, dtype=np.float32),
        np.atleast_1d(np.float32(gamma))
    ])
    x_t = torch.from_numpy(x).to(device).unsqueeze(0)  # (1,15)
    with torch.no_grad():
        c_hat_t, g_hat_t = model(x_t)
    c_hat = c_hat_t.squeeze(0).cpu().numpy()  # (7,)
    g_hat = g_hat_t.squeeze(0).cpu().numpy()  # (7,)

    # 原始预测 zonotope 中心 c_hat 和半径 g_hat
    qmin = c_hat - g_hat
    qmax = c_hat + g_hat

    # 物理可达范围：q_next = q + dq*dt + 0.5*a*dt^2，a∈[-acc_max, +acc_max]
    q_mid = np.asarray(q, dtype=np.float32) + np.asarray(dq, dtype=np.float32) * dt
    delta = 0.5 * np.asarray(acc_max, dtype=np.float32) * dt**2
    min_pos = q_mid - delta
    max_pos = q_mid + delta

    # 初步截断
    qmin_clamped = np.maximum(qmin, min_pos)
    qmax_clamped = np.minimum(qmax, max_pos)

    # 检查并修复冲突维度
    conflict_idxs = np.where(qmin_clamped > qmax_clamped)[0]
    if conflict_idxs.size > 0:
        print("Warning: after clamping, qmin > qmax on some joints. Fixing these joints to reachable bounds.")
        for i in conflict_idxs:
            print(f" Joint {i}: pred=[{qmin[i]:.3f},{qmax[i]:.3f}], "
                  f"reachable=[{min_pos[i]:.3f},{max_pos[i]:.3f}] -> "
                  f"clamped=[{min_pos[i]:.3f},{max_pos[i]:.3f}]")
            # 对冲突的关节仅用物理可达范围替换
            qmin_clamped[i] = min_pos[i]
            qmax_clamped[i] = max_pos[i]

    return qmin_clamped, qmax_clamped