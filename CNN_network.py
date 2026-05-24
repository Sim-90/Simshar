import torch
import torch.nn as nn
import torch.functional as F
from torchvision import models, datasets, transforms
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

#  making similar feature layers
class CNN_similar(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(CNN_similar,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16, kernel_size=(3,3),stride=(1,1), padding=(1,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32, kernel_size=(3,3),stride=(1,1), padding=(1,1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=(3,3),stride=(1,1), padding=(1,1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc1 = nn.Linear(32, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.reshape(x.shape[0],-1)
        # print(x.shape)
        x = self.fc1(x)
        return x

# batch_size =64

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


# # entire MNIST
# train_set = datasets.MNIST(root='dataset/', train=True, transform = transforms.ToTensor(), download = True)
# train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True)

# test_set = datasets.MNIST(root='dataset/', train=False, transform= transforms.ToTensor(), download= True)
# test_loader = DataLoader(dataset= test_set, batch_size=batch_size, shuffle=True)

# input_size=784
# num_classes = 10
# in_channels = 1
# learning_rate = 0.01
# num_epochs = 20
# batch_size = 64

# #  CNN model initialization
# model_sim = CNN_similar().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model_sim.parameters(), lr=learning_rate)

# # Training ...
# # train the CNN
# for epoch in range(num_epochs):
#     print(f'Epoch:{epoch}')
#     for batch_idx,(data,targets) in enumerate(train_loader):
#         data = data.to(device=device)
#         targets = targets.to(device=device)
#         # data = data.reshape(data.shape[0],-1) #flatten
#         # if epoch == 0 and batch_idx == 0:
#         #     print(data.shape)

#         scores = model_sim(data)
#         loss = criterion(scores, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# # Testing ...

# def check_accuracy(loader,model_sim):
#     if loader.dataset.train:
#         print(f"Accuracy on the training data")
#     else:
#         print(f"Accuracy on testing data")
#     num_correct = 0
#     num_samples = 0
#     model_sim.eval()

#     with torch.no_grad():
#         for x,y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
#             # x = x.reshape(x.shape[0],-1)
#             scores = model_sim(x)
#             _,predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)

#         print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

#     model_sim.train()
# check_accuracy(train_loader,model_sim)
# check_accuracy(test_loader, model_sim)

