import torch
import numpy as np
import pandas as pd
from Transformer import TransformerGamma
from tqdm import tqdm

# ---------- 配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 20                   # 每个样本采样点数
csv_path = './data/bounds_9m_gamma.csv'
num_samples = 1_000_000    # 读取前多少条

# ---------- 加载 γ 模型 ----------
transformer = TransformerGamma().to(device)
transformer.load_state_dict(
    torch.load("./models/transformer_gamma.pt", map_location=device)
)
transformer.eval()

@torch.no_grad()
def gamma_eval(q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
    """批量评估γ"""
    x = torch.cat([q, dq], dim=1).to(device)
    _, gamma = transformer(x)
    return gamma.squeeze()

# ---------- 数据加载与过滤 ----------
# 读取 CSV
df = pd.read_csv(csv_path, header=None, nrows=num_samples, low_memory=False)
# 跳过表头
if df.iloc[0].apply(lambda v: isinstance(v, str)).any():
    df = df.iloc[1:]
# 强制 float 并转数组
arr = df.astype(np.float32).values
# 原始 gamma 列索引
gamma_arr = arr[:, 28]
# 过滤掉 gamma < 2.5 的样本
mask = gamma_arr >= 2.5
filtered_count = mask.sum()
arr = arr[mask]
print(f"Filtered out {len(gamma_arr) - filtered_count} samples; {filtered_count} samples remain with gamma>=2.5.")

# 提取 c, g, dq
c = torch.tensor((arr[:, 14:28:2] + arr[:, 15:28:2]) / 2.0, dtype=torch.float32)
g = torch.tensor((arr[:, 15:28:2] - arr[:, 14:28:2]) / 2.0, dtype=torch.float32)
dq = torch.tensor(arr[:, 7:14], dtype=torch.float32)

# ---------- 计算 Cover Loss & 违规概率 ----------
total_cover = 0.0
count = 0
total_violations = 0
total_points = 0
batch_size = 256
num_samples = c.size(0)

for i in tqdm(range(0, num_samples, batch_size), desc='Compute Cover Loss'):
    c_batch = c[i:i+batch_size].to(device)
    g_batch = g[i:i+batch_size].to(device)
    dq_batch = dq[i:i+batch_size].to(device)
    batch = c_batch.size(0)
    # 采样 alpha
    alpha = torch.rand(batch, K, 7, device=device) * 2 - 1
    # 构造采样点
    q_p = c_batch.unsqueeze(1) + alpha * g_batch.unsqueeze(1)
    q_pf = q_p.view(-1, 7)
    dq_rep = dq_batch.unsqueeze(1).expand(-1, K, -1).reshape(-1, 7)
    # 评估 gamma
    gamma_p = gamma_eval(q_pf, dq_rep)
    # hinge loss
    cover = torch.relu(2.5 - gamma_p).mean().item()
    total_cover += cover * batch
    count += batch
    # 统计违规点
    violations = (gamma_p < 2.5).sum().item()
    total_violations += violations
    total_points += batch * K

# 平均 Cover Loss
avg_cover_loss = total_cover / count
# 违规概率 = 总违规采样点数 / 总采样点数
violation_prob = total_violations / total_points
print(f"Average Cover Loss over {count} samples: {avg_cover_loss:.4f}")
print(f"Probability of sampled points with gamma<2.5: {violation_prob:.4%}")
