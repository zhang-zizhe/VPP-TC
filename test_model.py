import os
import torch
import numpy as np
import pandas as pd
import random
from Transformer import TransformerGamma

# ---------- 配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './models/zonotope_net_1m2.pt'
csv_path = './data/bounds_9m_gamma.csv'
num_samples = 100_000   # 限制读取数据量
n_tests = 10000               # 随机选取测试样本数
K = 30                   # 每个样本采样点数

# ---------- 加载 ZonoNet 模型权重 ----------
class ZonotopeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(15, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.head_c = torch.nn.Linear(64, 7)
        self.head_g = torch.nn.Linear(64, 7)
        self.softplus = torch.nn.Softplus()
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return self.head_c(h), self.softplus(self.head_g(h)) + 1e-6

model = ZonotopeNet().to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ---------- γ 评分模型加载 ----------
transformer_model = TransformerGamma().to(device)
transformer_model.load_state_dict(
    torch.load("./models/transformer_gamma.pt", map_location=device)
)
transformer_model.eval()

@torch.no_grad()
def gamma_eval(q_prime: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
    x = torch.cat([q_prime, dq], dim=1).to(device)
    _, gamma = transformer_model(x)
    return gamma.squeeze()

# ---------- 读取并预处理数据 ----------
df = pd.read_csv(csv_path, header=None, nrows=num_samples, low_memory=False)
if df.iloc[0].apply(lambda v: isinstance(v, str)).any():
    df = df.iloc[1:]
arr = df.values.astype(np.float32)
gamma_arr = arr[:, 28]
mask = gamma_arr >= 2.5
print(f"Filtered out {(~mask).sum()} samples; {mask.sum()} remain.")
arr = arr[mask]
N = arr.shape[0]

# 提取原始数据
q_all = arr[:, 0:7]
dq_all = arr[:, 7:14]
gamma_all = arr[:, 28]
# 真实区间
qmin_all = arr[:, 14:28:2]
qmax_all = arr[:, 15:28:2]

# 随机选样索引
indices = random.sample(range(N), n_tests)
count = 0
contain = 0
acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)
# acc_max_t = torch.tensor(acc_max, device=device).unsqueeze(0)
# ---------- 测试循环 ----------
for idx in indices:
    # 原始样本
    q = q_all[idx]
    dq = dq_all[idx]
    gamma = gamma_all[idx]
    qmin_true = qmin_all[idx]
    qmax_true = qmax_all[idx]

    x   = torch.tensor(np.concatenate([q, dq, [gamma]]), dtype=torch.float32, device=device).unsqueeze(0)

    # 网络预测 c, g
    c_hat, g_hat = model(x)
    c_hat = c_hat.cpu().detach().numpy().ravel()
    g_hat = g_hat.cpu().detach().numpy().ravel()
    qmin_hat = c_hat - g_hat
    qmax_hat = c_hat + g_hat
    q_mid = q +dq * 0.02
    delta = 0.5 * acc_max * (0.02**2)                         # (1,7)
    q_neg = q_mid - delta                                     # 最小可达
    q_pos = q_mid + delta                                     # 最大可达
    if np.any(q_neg > qmax_hat) or np.any(q_pos < qmin_hat):
        # print(f"Sample {idx} skipped due to out of bounds prediction.")
        contain += 1
        # continue

    # 打印对比
    # print(f"Sample {idx}:")
    # print(f"  q       = {q}")
    # print(f"  dq      = {dq}")
    # print(f"  True qmin: {qmin_true}")
    # print(f"  Pred qmin: {qmin_hat}")
    # print(f"  Δ qmin   : {qmin_hat - qmin_true}")
    # print(f"  True qmax: {qmax_true}")
    # print(f"  Pred qmax: {qmax_hat}")
    # print(f"  Δ qmax   : {qmax_hat - qmax_true}")
    # print(f"  q_mid    = {q_mid}")

    # 在预测 zonotope 内采样测试
    alphas = torch.rand(K, 7, device=device) * 2 - 1
    qps = torch.tensor(c_hat, device=device).unsqueeze(0) + alphas * torch.tensor(g_hat, device=device).unsqueeze(0)
    dq_rep = torch.tensor(dq, device=device).unsqueeze(0).expand(K, -1)
    gamma_ps = gamma_eval(qps, dq_rep)
    viol = (gamma_ps < 2.5).sum().item()
    if viol > 0:
        count += 1
        # print(f"Sample {idx}:")
        # print(f"  Violation rate: {viol}/{K} = {viol/K:.2%}\n")
        # print(f"count = {count}, total samples = {n_tests}")
print(f"Total violations: {count} out of {n_tests} samples.")
print(f"Containment violations: {contain} out of {n_tests} samples.")
print("Test complete.")
