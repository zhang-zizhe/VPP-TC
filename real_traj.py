import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------- 文件路径 ----------
CSV_PATH = Path("trajectories.csv")
MODEL_PATH = Path("../../PythonProject3/torque-constraint-learning/transformer_gamma.pt")

# ---------- 加载数据 ----------
df = pd.read_csv(CSV_PATH)
q = df.iloc[:, 2:9].values.astype(np.float32)    # 7维
qd = df.iloc[:, 9:16].values.astype(np.float32)  # 7维
X = np.concatenate([q, qd], axis=1)

# ---------- 定义模型 ----------
import torch.nn as nn
class MLPGamma(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 2)  # γ1 和 γ2
        )

    def forward(self, q):
        gamma_logits = self.net(q)
        γ1, γ2 = gamma_logits[:, 0], gamma_logits[:, 1]
        Γ = γ1 - γ2
        return gamma_logits, Γ
class TransformerGamma(nn.Module):
    def __init__(self, input_dim=14, d_model=64, nhead=2, num_layers=4, dropout=0.1):
        super().__init__()
        self.linear_encoder = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.linear_encoder(x)
        x = x + self.positional_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        gamma = logits[:, 0] - logits[:, 1]
        return logits, gamma

# ---------- 加载模型 ----------
device = torch.device("cpu")
model = TransformerGamma().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------- 推理 ----------
@torch.no_grad()
def predict_gamma(x_batch):
    x_tensor = torch.tensor(x_batch, dtype=torch.float32).to(device)
    _, gamma = model(x_tensor)
    return gamma.cpu().numpy()

BATCH = 1024
gamma_values = []
for i in tqdm(range(0, len(X), BATCH), desc="Predicting Γ"):
    gamma_values.append(predict_gamma(X[i:i+BATCH]))
gamma_values = np.concatenate(gamma_values)

# ---------- 评估 ----------
y_pred = (gamma_values < 2.5).astype(int)  # γ<3 认为是碰撞（1类）
y_true = np.zeros_like(y_pred)            # 真实轨迹为安全（0类）

accuracy = (y_pred == y_true).mean()
false_collision_count = (y_pred == 1).sum()

print(f"Accuracy：{accuracy*100:.2f}%")
print(f"misjudged as collisions：{false_collision_count} / {len(X)}")
df_first_traj = df[df["traj_idx"] == 30]
q_first  = df_first_traj.iloc[:, 2:9].values.astype(np.float32)
qd_first = df_first_traj.iloc[:, 9:16].values.astype(np.float32)
X_first  = np.concatenate([q_first, qd_first], axis=1)

# ---------- 预测 Γ ----------
gamma_first_traj = predict_gamma(X_first)

# ---------- 打印预测值和是否误判 ----------
print("\n🧪 第一个轨迹（traj_idx=0）预测结果：")
for i, gamma in enumerate(gamma_first_traj):
    flag = "✅ Safe" if gamma >= 2.5 else "❌ Collision"
    print(f"  第{i}帧：Γ = {gamma:.4f} → {flag}")

# ---------- 总体统计 ----------
n_total = len(gamma_first_traj)
n_safe = np.sum(gamma_first_traj >= 2.5)
n_coll = n_total - n_safe
print(f"\n📊 轨迹总帧数：{n_total}，被判为安全：{n_safe}，被误判为碰撞：{n_coll}")

# ---------- 保存50条轨迹的统计信息 ----------
results = []

for traj_id in range(50):
    df_traj = df[df["traj_idx"] == traj_id]
    q_traj  = df_traj.iloc[:, 2:9].values.astype(np.float32)
    qd_traj = df_traj.iloc[:, 9:16].values.astype(np.float32)
    X_traj  = np.concatenate([q_traj, qd_traj], axis=1)
    gamma_traj = predict_gamma(X_traj)

    n_total = len(gamma_traj)
    n_safe  = np.sum(gamma_traj >= 2.5)
    n_coll  = n_total - n_safe

    results.append({
        "traj_idx": traj_id,
        "total_frames": n_total,
        "safe_frames": n_safe,
        "collision_frames": n_coll
    })

# ---------- 写入 CSV ----------
results_df = pd.DataFrame(results)
results_df.to_csv("trajectory_gamma_stats.csv", index=False)
print("✅ 已保存轨迹统计到 trajectory_gamma_stats.csv")
