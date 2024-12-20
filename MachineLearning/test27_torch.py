import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 신경망 클래스 정의
class NeuralNetwork(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(NeuralNetwork, self).__init__()
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # 은닉층과 출력층 정의
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 데이터 로드
training_data = np.loadtxt('./mnist_data/mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('./mnist_data/mnist_test.csv', delimiter=',', dtype=np.float32)

print("training_data.shape = ", training_data.shape, " ,  test_data.shape = ", test_data.shape)
print("training_data[0,0] = ", training_data[0,0], ",  test_data[0,0] = ", test_data[0,0])
print("len(training_data[0]) = ", len(training_data[0]), ",  len(test_data[0]) = ", len(test_data[0]))

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
epochs = 1

# 신경망 초기화
nn_model = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
criterion = nn.BCELoss()
optimizer = optim.SGD(nn_model.parameters(), lr=learning_rate)

start_time = datetime.now()

# 학습
for epoch in range(epochs):

    for step in range(len(training_data)):
        target_data = np.zeros(output_nodes) + 0.01    
        target_data[int(training_data[step, 0])] = 0.99
        input_data = ((training_data[step, 1:] / 255.0) * 0.99) + 0.01
        
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(target_data, dtype=torch.float32).unsqueeze(0)
        
        optimizer.zero_grad()
        output = nn_model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        
        if step % 400 == 0:
            print("step = ", step,  ",  loss_val = ", loss.item())

end_time = datetime.now()
print("\nelapsed time = ", end_time - start_time)

# 정확도 측정
def accuracy(nn_model, test_data):
    matched_list = []
    not_matched_list = []
    
    for index in range(len(test_data)):
        label = int(test_data[index, 0])
        data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01
        input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        
        output = nn_model(input_tensor)
        predicted_num = torch.argmax(output).item()
        
        if label == predicted_num:
            matched_list.append(index)
        else:
            not_matched_list.append(index)
    
    print("Current Accuracy = ", 100 * (len(matched_list) / len(test_data)), " %")
    return matched_list, not_matched_list

accuracy(nn_model, test_data)
