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