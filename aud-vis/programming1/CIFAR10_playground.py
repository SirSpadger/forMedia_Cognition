import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from typing import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

################### 数据集初始化与读入 ###################
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])
class CustomCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root,
        train,
        transform = None,
        target_transform = None,
        download = False,
    ):
        super().__init__(root, train, transform, target_transform, download)
        new_data = []
        new_target = []
        print(len(self.targets))
        for i in tqdm(range(len(self.data))):
            img, target = self.data[i], self.targets[i]
            # print(self.data.shape)
            # print(img.shape)
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
            
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            # self.data[i], self.targets[i] = img, target

            new_data.append(img)
            new_target.append(target)
        
        self.data = torch.Tensor(np.stack(new_data, axis=0)).to(device)
        self.targets = torch.Tensor(new_target).to(torch.int64).to(device)
        # self.targets = new_target
        print(len(self.targets))
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

# train_dset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=True,download=False,transform=train_transform)
# test_dset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False,download=False,transform=transforms.ToTensor())
train_dset = CustomCIFAR10(root='./CIFAR10',train=True,download=False,transform=train_transform)
test_dset = CustomCIFAR10(root='./CIFAR10',train=False,download=False,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=128, shuffle=False, num_workers=0)
#######################################################


################### 构建模型 ###################
class Net(nn.Module):
    def __init__(self,act):
        super(Net, self).__init__()
        # 卷积层 (32x32x3的图像)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 卷积层(16x16x16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 卷积层(8x8x32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        if act == 'relu':
            self.act = F.relu
        elif act == 'tanh':
            self.act = F.tanh
        elif act == 'sigmoid':
            self.act = F.sigmoid

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)

        x = self.act(self.fc1(x))

        x = self.fc2(x)
        return x
#######################################################

################### 模型加入batchnorm ###################
class BnNet(nn.Module):
    def __init__(self, act):
        super(BnNet, self).__init__()
        # 卷积层 (32x32x3的图像)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # 卷积层(16x16x16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 卷积层(8x8x32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)

        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.bn4 = nn.BatchNorm1d(500)

        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)

        if act == 'relu':
            self.act = F.relu
        elif act == 'tanh':
            self.act = F.tanh
        elif act == 'sigmoid':
            self.act = F.sigmoid

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))

        # flatten image input
        x = x.view(-1, 64 * 4 * 4)

        x = self.act(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x
################### 构建模型 ###################



class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        ################### 代码填空：请在此填补模型定义代码 ###################
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, 1, padding=0)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.act = F.tanh
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        ##################################################################

    def forward(self, x):
        # convolutional layers
        ################### 代码填空：请在此填补前向计算代码 ###################
        x = self.act(self.conv1(x))
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))
        x = self.pool(self.act(self.conv4(x)))
        x = self.act(self.conv5(x))
        x = self.avgpool(self.act(self.conv6(x)))
        x = x.view(-1, 512)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        ##################################################################
        return x

class BnDeepNet(nn.Module):
    def __init__(self,act):
        super(BnDeepNet, self).__init__()
        ################### 代码填空：请在此填补模型定义代码 ###################
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 1, padding=0)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 256)
        self.bnfc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bnfc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        if act == 'relu':
            self.act = F.relu
        elif act == 'tanh':
            self.act = F.tanh
        elif act == 'sigmoid':
            self.act = F.sigmoid
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        ###################################################################

    def forward(self, x):
        # convolutional layers
        ################### 代码填空：请在此填补前向计算代码 ###################
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(self.act(self.bn4(self.conv4(x))))
        x = self.act(self.bn5(self.conv5(x)))
        x = self.avgpool(self.act(self.bn6(self.conv6(x))))
        x = x.view(-1, 512)

        x = self.act(self.bnfc1(self.fc1(x)))
        x = self.act(self.bnfc2(self.fc2(x)))
        x = self.act(self.fc3(x))
        return x
        ##################################################################


################### 训练前准备 ###################


model = BnDeepNet('relu').to(device)
criterion = nn.CrossEntropyLoss().to(device)

optimizer_type = "SGD" #或者换成AdamW
if optimizer_type == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=0.001)
elif optimizer_type == "AdamW":
    ########## 代码填空：请在此填补Adam优化器计算代码, lr=0.0001 ###########
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    ##################################################################
    pass

n_epochs = 100
train_losses = []
valid_losses = []
accuracies = []
################################################


################### 训练+验证 ###################
for epoch in range(n_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    model.train()
    for idx,(img,label) in tqdm(enumerate(train_loader)):
        # img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * img.shape[0]

    model.eval()
    correct = 0
    total = 0
    for idx,(img,label) in tqdm(enumerate(test_loader)):
        # img, label = img.to(device), label.to(device)
        output = model(img)
        loss = criterion(output, label)
        valid_loss += loss.item() * img.shape[0]
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    
    train_loss = train_loss / len(train_dset)
    valid_loss = valid_loss / len(test_dset)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    accuracy = correct / total
    accuracies.append(accuracy)

    print(f"Epoch:{epoch}, Acc:{correct/total}, Train Loss:{train_loss}, Test Loss:{valid_loss}")
################################################

################### 曲线绘制 ###################
print("MAX ACC: ",np.max(accuracies))
plt.plot(range(n_epochs), train_losses, label='Train Loss')
plt.plot(range(n_epochs), valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Loss.png")
plt.clf()
# 绘制验证集准确率随epoch的变化曲线
plt.plot(range(n_epochs), accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("Acc.png")
################################################