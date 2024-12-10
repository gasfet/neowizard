import torch

x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6,1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6,1)

print(x_train)
print(y_train)
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.x_train.shape[0]
dataset = CustomDataset(x_train, y_train)

train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)
total_batch = len(train_loader)

print(total_batch)

from torch import nn

class MyLinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, data):
        prediction = self.linear_stack(data)

        return prediction
model = MyLinearRegressionModel()        
loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
for epoch in range(2):

    for idx, batch_data in enumerate(train_loader):

        x_train_batch, y_train_batch = batch_data

        output_batch = model(x_train_batch)

        print('==============================================')
        print('epoch =', epoch+1, ', batch_idx =', idx+1, ',',
              len(x_train_batch), len(y_train_batch), len(output_batch))
        print('==============================================')

        loss = loss_function(output_batch, y_train_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()