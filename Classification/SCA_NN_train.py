import torch
import scipy.io
import mat73
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
#
# mat = scipy.io.loadmat('data_SCA_classification.mat')
mat = scipy.io.loadmat('data_SCA_biclass.mat')
# mat = mat73.loadmat('data_SCA_classification.mat')

# print(mat['data_SCA_classification'].shape)
print(mat['data_SCA_biclass'].shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (torch.cuda.is_available()==True):
    print('GPU is available on this computer!')
else:
    print('GPU is not available on this computer!')


# Generate sample data
# x = mat['data_SCA_classification'][:, 0:7]
# # y = mat['data_SCA_classification'][:, 7:8]
# y = mat['data_SCA_classification'][:, 7]
x = mat['data_SCA_biclass'][:, 0:7]
y = mat['data_SCA_biclass'][:, 7:9]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and move to GPU
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define a neural network model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.linear1 = nn.Linear(7, 80)
        self.linear2 = nn.Linear(80, 50)
        self.linear3 = nn.Linear(50, 30)
        self.linear4 = nn.Linear(30, 10)
        self.linear5 = nn.Linear(10, 2)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.linear1(x)
        out = self.leakyrelu(out)
        out = self.linear2(out)
        out = self.leakyrelu(out)
        out = self.linear3(out)
        out = self.leakyrelu(out)
        out = self.linear4(out)
        out = self.leakyrelu(out)
        out = self.linear5(out)
        return out


# Create an instance of the model and move to GPU
model = BinaryClassifier().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.RMSprop(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Create DataLoader for training and validation sets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=1024)

# Train the model
num_epochs = 2000
loss_values = []
accuracy_values = []
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

        # Calculate training accuracy
        correct += (outputs.argmax(1) == targets[:, 1]).sum().item()
        total += targets.size(0)

    train_loss /= len(train_loader.dataset)
    train_accuracy = correct / total

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            correct += (outputs.argmax(1) == targets[:, 1]).sum().item()
            total += targets.size(0)

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / total

    loss_values.append(val_loss)
    accuracy_values.append(val_accuracy)

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        # Save the model
        path = f'./NN_result/SCA_boundary_{epoch + 1}.pt'
        torch.save(model, path)


# print('Model saved successfully!')

# Convert tensors back to numpy arrays and move to CPU
x_np = x_train_tensor.cpu().detach().numpy()
y_np = model(x_train_tensor).cpu().detach().numpy()

# # Plot the data and decision boundaries
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train[:, 0], cmap='bwr', label='Output 1')
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train[:, 1], cmap='coolwarm', label='Output 2')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Training Data')
# plt.legend()
# plt.show()

# Plot the loss and accuracy curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_values)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(accuracy_values)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')

plt.tight_layout()
plt.show()
