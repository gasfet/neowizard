import torch
import torch.nn as nn
import torch.optim as optim

# LogicGate Class using PyTorch
class LogicGatePyTorch(nn.Module):
    def __init__(self):
        super(LogicGatePyTorch, self).__init__()
        
        # Hidden layer (2 inputs -> 6 hidden units)
        self.hidden = nn.Linear(2, 6)
        # Output layer (6 hidden units -> 1 output)
        self.output = nn.Linear(6, 1)
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        z2 = self.hidden(x)  # Hidden layer linear combination
        a2 = self.relu(z2)  # 이곳에 sigmoid를 사용하면 XOR문제를 해결하지 못한다. 즉 ReLU를 사용하면 문제 해결이 된다. 
        z3 = self.output(a2)  # Output layer linear combination
        y = self.sigmoid(z3)  # Output layer activation
        return y

# Input and target data
xdata = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

Gate = LogicGatePyTorch()
criterion = nn.BCELoss()
optimizer = optim.SGD(Gate.parameters(), lr=0.02)
#Tdata = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32)
#Tdata = torch.tensor([[0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)
Tdata = torch.tensor([[1.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
#Tdata = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

for epoch in range(8001):
    optimizer.zero_grad()  # Zero the gradients
    y_pred = Gate(xdata)  # Forward pass

    loss1 = criterion(y_pred, Tdata)  # Compute loss
    loss1.backward()  # Backward pass
    optimizer.step()  # Update weights

    if epoch % 400 == 0:
        print(f"Epoch {epoch}, error value = {loss1.item()}")

# Prediction function
def predict(model, input_data):
    with torch.no_grad():
        y_pred = model(input_data)
        logical_val = (y_pred > 0.5).float()
        return y_pred, logical_val

# Test the trained model

test_data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

print("GATE")
for input_data in test_data:
    sigmoid_val, logical_val = predict(Gate, input_data.unsqueeze(0))
    print(input_data.numpy(), "=", sigmoid_val.item(), ",", logical_val.item())
