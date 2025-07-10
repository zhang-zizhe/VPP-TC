import torch

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

import numpy as np

# ---------- 配置 ----------
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------- 全局物理参数 ----------
dt        = 0.02
acc_max   = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)
acc_max_t = torch.tensor(acc_max, device=device).unsqueeze(0)

class ZonotopeNetC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1    = torch.nn.Linear(14, 128)
        self.fc2    = torch.nn.Linear(128, 128)
        self.fc3    = torch.nn.Linear(128, 64)
        self.head_c = torch.nn.Linear(64, 7)
        self.head_g = torch.nn.Linear(64, 7)

    def forward(self, x):
        # x: (B,14) = [q(7), dq(7)]
        q     = x[:, :7]
        dq    = x[:, 7:14]
        q_mid = q + dq * dt                             # (B,7)
        delta = 0.5 * acc_max_t * (dt**2)               # broadcast to (B,7)

        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))

        c_tilde = self.head_c(h)                        # ∈ ℝ^(B×7)
        g_tilde = self.head_g(h)                        # ∈ ℝ^(B×7)

        # 1) c_hat ∈ [q_mid - δ, q_mid + δ]
        c_hat = q_mid + delta * torch.tanh(c_tilde)

        # 2) 动态上界：保证 c_hat±g_hat 不越出 [q_mid-δ, q_mid+δ]
        bound_lower = c_hat - (q_mid - delta)            # = c_hat - q_min
        bound_upper = (q_mid + delta) - c_hat            # = q_max - c_hat
        g_bound     = torch.min(bound_lower, bound_upper).clamp(min=1e-6)

        # 3) g_hat ∈ [0, g_bound]
        g_hat = g_bound * torch.sigmoid(g_tilde)

        return c_hat, g_hat
# class ZonotopeNetC(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1    = torch.nn.Linear(14, 128)
#         self.fc2    = torch.nn.Linear(128, 128)
#         self.fc3    = torch.nn.Linear(128, 64)
#         self.head_c = torch.nn.Linear(64, 7)
#         self.head_g = torch.nn.Linear(64, 7)
#         self.softplus = torch.nn.Softplus()

#     def forward(self, x):
#         h = torch.relu(self.fc1(x))
#         h = torch.relu(self.fc2(h))
#         h = torch.relu(self.fc3(h))
#         c_raw = self.head_c(h)                  # 原始中心
#         g_raw = self.softplus(self.head_g(h))   # 保证非负
#         return c_raw, g_raw