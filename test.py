import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score
import torch.nn as nn

# ---------- 数据集 ----------
LABEL_COL = 21

class CollisionDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.iloc[:, :14].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, LABEL_COL].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------- 模型结构 ----------
class TransformerGamma(nn.Module):
    def __init__(self, input_dim=14, d_model=128, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.linear_encoder = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, q):
        q = q.unsqueeze(-1)
        x = self.linear_encoder(q)
        x = x + self.positional_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)
        gamma_logits = self.classifier(x)
        γ1, γ2 = gamma_logits[:, 0], gamma_logits[:, 1]
        Γ = γ1 - γ2
        return gamma_logits, Γ
# ---------- 2. MLP Model ----------
class MLPGamma(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # γ1 和 γ2
        )

    def forward(self, q):
        gamma_logits = self.net(q)
        γ1, γ2 = gamma_logits[:, 0], gamma_logits[:, 1]
        Γ = γ1 - γ2
        return gamma_logits, Γ
# ---------- 评估函数 ----------
@torch.no_grad()
def evaluate_with_fixed_gamma(model, loader, device, threshold):
    model.eval()
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        _, gamma = model(X)
        preds = (gamma < threshold).long()
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, pos_label=1)

    return acc, recall

# ---------- 主流程 ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerGamma().to(device)
    model.load_state_dict(torch.load("transformer_gamma.pt", map_location=device))
    model.eval()

    datasets = {
        "Dataset 1": "collision_results_1_limit_sampling.csv",
        "Dataset 2": "collision_results_2_limit_sampling.csv",
    }

    for name, path in datasets.items():
        ds = CollisionDataset(Path(path))
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        acc, recall = evaluate_with_fixed_gamma(model, loader, device, threshold=3)
        print(f"\n📊 {name}:")
        print(f"Accuracy = {acc*100:.2f}%")
        print(f"Recall   = {recall*100:.2f}%")

if __name__ == "__main__":
    main()
