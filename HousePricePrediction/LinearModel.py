import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

boston_data = pd.read_csv("D:\Coderbase\HousePricePrediction\BostonHousing.csv")

# 1. Load the dataset manually
x = boston_data.drop('medv', axis=1)
y = boston_data['medv']

x = np.array(x)
y = np.array(y)

# Split dataset manually (80% train, 20% test)
n_samples = x.shape[0]
train_size = int(0.8 * n_samples)
X_train, X_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 2. Normalize the data (standardization) manually
mean = X_train.mean(0)
std = X_train.std(0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 3. Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # single linear layer
    
    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]  # Number of features
model = LinearRegressionModel(input_dim)

# 4. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Train the Model
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train)
    
    # Compute loss
    loss = criterion(y_pred, y_train)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 100 epochs
    if (epoch) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 6. Evaluate the Model on Test Data
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss = criterion(y_test_pred, y_test)
    for i in range(len(y_test_pred)):
        print(f"y_eval: {y_test_pred[i].item()}, y_test: {y_test[i]}, diff: {y_test_pred[i].item() - y_test[i]}.")
    print(f'Test Loss: {test_loss.item():.4f}')

# 7. Optional: Calculate R-squared
def r2_score(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()

r2 = r2_score(y_test, y_test_pred)
print(f'R-squared: {r2:.4f}')
