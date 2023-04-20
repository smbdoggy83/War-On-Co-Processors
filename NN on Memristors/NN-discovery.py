# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:40:18 2023

@author: Seth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# Define the input and output dimensions
input_dim = 2
output_dim = 1

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# Create the neural network object and define the loss function and optimizer
model = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define the input and target data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y = torch.tensor([0, 1, 1, 0], dtype=torch.float).view(-1, 1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the neural network
num_epochs = 50
batch_size = 32
num_batches = int(np.ceil(len(X_train) / batch_size))

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(num_batches):
        # Get the batch data
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(X_train))
        batch_X = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    # Print the loss for this epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

# Evaluate the neural network on the test set
outputs = model(X_val)
predicted = (outputs > 0.5).float()
accuracy = (predicted == y_val).float().mean()
print('Validation accuracy:', accuracy.item())
