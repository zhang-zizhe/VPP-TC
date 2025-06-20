import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Transformer import TransformerGamma

# ---------- 设备配置 & CUDA 优化 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
torch.backends.cudnn.benchmark = True

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

# ---------- 数据集定义（不做归一化） ----------
class ZonotopeDataset(Dataset):
    def __init__(self, q, dq, gamma, c, g):
        self.q     = torch.tensor(q,     dtype=torch.float32)
        self.dq    = torch.tensor(dq,    dtype=torch.float32)
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.c     = torch.tensor(c,     dtype=torch.float32)
        self.g     = torch.tensor(g,     dtype=torch.float32)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        x = torch.cat([self.q[idx], self.dq[idx], self.gamma[idx]], dim=0)
        return x, self.c[idx], self.g[idx]

# ---------- 模型结构 ----------
class ZonotopeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(15, 128)
        self.fc2     = nn.Linear(128, 128)
        self.fc3     = nn.Linear(128, 64)
        self.head_c  = nn.Linear(64, 7)
        self.head_g  = nn.Linear(64, 7)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        c_hat = self.head_c(h)
        g_hat = self.softplus(self.head_g(h)) + 1e-6
        return c_hat, g_hat

# ------- 超参数 -------
batch_size   = 128
lr           = 1e-3
epochs       = 40
K            = 80
w_cover, w_contain, w_vol = 1.0, 1.0, 0.0
num_samples  = 5_000_000
dt = 0.02
acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)
acc_max_t = torch.tensor(acc_max, device=device).unsqueeze(0)
csv_path     = './data/bounds_9m_gamma.csv'

# ------ 读取并预处理数据 ------
preview = pd.read_csv(csv_path, header=None, nrows=3, low_memory=False)
skiprows = 1 if preview.iloc[0].apply(lambda v: isinstance(v, str)).any() else 0

df = pd.read_csv(
    csv_path, header=None, skiprows=skiprows,
    nrows=int(num_samples), low_memory=False
)
arr = df.values.astype(np.float32)

# 只保留 gamma >= 2.5 的样本
gamma_arr = arr[:, 28]
mask = gamma_arr >= 2.5
print(f"Filtered out {(~mask).sum()} samples; {mask.sum()} remain.")
arr = arr[mask]

# 拆分原始特征和标签
q_all     = arr[:, 0:7]
dq_all    = arr[:, 7:14]
gamma_all = arr[:, 28:29]         # shape (N,1)
mins      = arr[:, 14:28:2]
maxs      = arr[:, 15:28:2]
c_all     = (mins + maxs) / 2.0
g_all     = (maxs - mins) / 2.0

# 划分训练/验证集（此处不做归一化）
train_q, val_q, train_dq, val_dq, train_gam, val_gam, train_c, val_c, train_g, val_g = \
    train_test_split(
        q_all, dq_all, gamma_all, c_all, g_all,
        test_size=0.1, random_state=42, shuffle=True
    )

# DataLoader
dataset_train = ZonotopeDataset(train_q, train_dq, train_gam, train_c, train_g)
dataset_val   = ZonotopeDataset(val_q,   val_dq,   val_gam,   val_c,   val_g)
loader_train  = DataLoader(dataset_train, batch_size=batch_size,
                           shuffle=True,  pin_memory=True, num_workers=4)
loader_val    = DataLoader(dataset_val,   batch_size=batch_size,
                           shuffle=False, pin_memory=True, num_workers=2)

# ---------- 模型/优化器/学习率调度 ----------
model     = ZonotopeNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
mseloss   = nn.MSELoss()
scaler    = torch.cuda.amp.GradScaler()

# ------ 训练主循环 ------
for epoch in range(1, epochs + 1):
    if epoch < 5:
        w_contain = 10.0
    else:
        w_contain = 1.0
    print(f"Epoch {epoch}/{epochs}, LR={scheduler.get_last_lr()[0]:.1e}")
    model.train()
    accum = {'reg':0,'cover':0,'contain':0,'vol':0}
    violate_count = 0
    total_samples = 0

    for x, c_gt, g_gt in tqdm(loader_train, desc="Training", leave=False):
        x, c_gt, g_gt = x.to(device), c_gt.to(device), g_gt.to(device)
        # 直接使用原始尺度 q, dq
        q   = x[:, :7]
        dq  = x[:, 7:14]
        # 计算 q_mid, q_neg, q_pos
        q_mid = q + dq * dt                                      # (batch,7)
        delta = 0.5 * acc_max_t * (dt**2)                         # (1,7)
        q_neg = q_mid - delta                                     # 最小可达
        q_pos = q_mid + delta                                     # 最大可达
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            c_hat, g_hat = model(x)
            # 回归损失
            loss_reg = 2*mseloss(c_hat, c_gt) + mseloss(g_hat, g_gt)
            # 覆盖损失
            alpha     = torch.rand(x.size(0), K, 7, device=device)*2 - 1
            q_p       = c_hat.unsqueeze(1) + alpha * g_hat.unsqueeze(1)
            q_pf      = q_p.reshape(-1, 7)
            dq_rep    = dq.unsqueeze(1).expand(-1, K, -1).reshape(-1, 7)
            gamma_p   = gamma_eval(q_pf, dq_rep)
            loss_cover  = torch.relu(2.6 - gamma_p).mean()
            violate_count += (gamma_p < 2.6).sum().item()
            total_samples += gamma_p.numel()
            # 包含性损失
            qmin_hat = c_hat - g_hat
            qmax_hat = c_hat + g_hat
            loss_lower = torch.relu(q_neg - qmax_hat).mean()
            loss_upper = torch.relu(qmin_hat - q_pos).mean()
            loss_contain = loss_lower + loss_upper
            # 体积正则
            loss_vol   = (g_hat**2).sum(dim=1).mean()
            # 总损失
            loss = 2*loss_reg + w_cover*loss_cover + w_contain*loss_contain + w_vol*loss_vol

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accum['reg']     += loss_reg.item()   * x.size(0)
        accum['cover']   += loss_cover.item() * x.size(0)
        accum['contain'] += loss_contain.item()* x.size(0)
        accum['vol']     += loss_vol.item()   * x.size(0)

    # 统计并输出训练指标
    for k in accum:
        accum[k] /= len(dataset_train)
    print(f" Train losses: reg={accum['reg']:.4f}, cover={accum['cover']:.4f}, contain={accum['contain']:.4f}, vol={accum['vol']:.6f}")
    print(f" Train violation rate: {violate_count/total_samples:.4f}")

    # ------ 验证 ------
    model.eval()
    val_accum = {'reg':0,'cover':0,'contain':0,'vol':0}
    val_count = 0
    with torch.no_grad():
        for x, c_gt, g_gt in loader_val:
            x, c_gt, g_gt = x.to(device), c_gt.to(device), g_gt.to(device)
            batch = x.size(0)
            q_mid_val = x[:, :7] + x[:, 7:14] * dt
            delta     = 0.5 * acc_max_t * (dt**2)
            q_neg_val = q_mid_val - delta
            q_pos_val = q_mid_val + delta


            c_hat, g_hat = model(x)
            loss_reg = mseloss(c_hat, c_gt) + mseloss(g_hat, g_gt)
            # 覆盖
            alphas = torch.rand(batch, K, 7, device=device)*2 - 1
            q_p    = c_hat.unsqueeze(1) + alphas * g_hat.unsqueeze(1)
            q_pf   = q_p.view(-1,7)
            dq_rep = x[:,7:14].unsqueeze(1).expand(-1, K, -1).reshape(-1,7)
            loss_cover  = torch.relu(2.6 - gamma_eval(q_pf, dq_rep)).mean()
            # 包含
            qmin_hat = c_hat - g_hat
            qmax_hat = c_hat + g_hat

            loss_lower = torch.relu(q_neg_val - qmax_hat).mean()
            loss_upper = torch.relu(qmin_hat - q_pos_val).mean()
            loss_contain = loss_lower + loss_upper
            # 体积
            loss_vol   = (g_hat**2).sum(dim=1).mean()

            val_accum['reg']     += loss_reg.item()   * batch
            val_accum['cover']   += loss_cover.item() * batch
            val_accum['contain'] += loss_contain.item()* batch
            val_accum['vol']     += loss_vol.item()   * batch
            val_count += batch

    for k in val_accum:
        val_accum[k] /= val_count
    print(f" Val losses: reg={val_accum['reg']:.4f}, cover={val_accum['cover']:.4f}, contain={val_accum['contain']:.4f}, vol={val_accum['vol']:.6f}")

    scheduler.step()

# 保存整个模型
os.makedirs("./models", exist_ok=True)
torch.save(model.state_dict(), "./models/zonotope_net_5m.pt")
print("Model saved.")
