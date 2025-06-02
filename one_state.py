import torch
import numpy as np
import torch.nn as nn

# --------------------------
# 指定关节状态（q 和 qd）
# --------------------------
q = [
    1.3721778,
    0.9264038,
   -0.39804775,
   -2.7748284,
    2.4984193,
    3.1535788,
   -0.8558559
]



qd = [
    0.17932661,
   -0.15019508,
    0.026845306,
   -0.20090489,
    0.04449944,
    0.16816235,
   -0.027525447
]


x = np.concatenate([q, qd])[None, :]  # (1, 14)

# --------------------------
# 模型定义
# --------------------------
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
        x = x.unsqueeze(-1)  # (B, 14, 1)
        x = self.linear_encoder(x)  # (B, 14, d_model)
        x = x + self.positional_encoding.unsqueeze(0)  # 加位置编码
        x = self.transformer(x)  # Transformer 编码
        x = x.mean(dim=1)  # 池化
        logits = self.classifier(x)  # 输出 (B, 2)
        gamma = logits[:, 0] - logits[:, 1]
        return logits, gamma

# --------------------------
# 加载模型并推理
# --------------------------
device = torch.device("cpu")
model = TransformerGamma().to(device)
model.load_state_dict(torch.load("transformer_gamma.pt", map_location=device))
model.eval()

x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
_, gamma = model(x_tensor)

# --------------------------
# 输出预测结果
# --------------------------
gamma_val = gamma.item()
flag = "✅ Safe" if gamma_val >= 2.5 else "❌ Collision"
print(f"预测结果：Γ = {gamma_val:.4f} → {flag}")
