import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)




class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = nn.Linear(1352,128)
        self.layer2 = nn.Linear(128,10)


    def forward(self, x):

        x = self.conv(x)
        x = torch.relu(self.pool(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)

        return x


model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)


for epoch in range(10):

    total_loss = 0

    for X_batch, Y_batch in train_loader:

        output = model(X_batch)
        loss = loss_fn(output, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f" Epoch: {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}")



# Evaluation

correct = 0
total = 0

with torch.no_grad():

    for X_batch, Y_batch in train_loader:

       output = model(X_batch)
       predictions = torch.argmax(output, dim = 1)

       correct += (predictions == Y_batch).sum().item()
       total += Y_batch.size(0)

print(f"Accuracy: {correct/total*100:.2f}%")
       


