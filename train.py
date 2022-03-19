import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from ResNet50_improve import *


def train_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            img, labels = data
            img, labels = img.to(device), labels.to(device)
            d = img.size()
            e = labels.size()
            out = model(img)
            f = out.size()
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('Accuracy of the network on the train image: %d %%' % (100 * correct / total))
    return 100.0 * correct / total


def test_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            img, labels = data
            img, labels = img.to(device), labels.to(device)
            d = img.size()
            e = labels.size()
            out = model(img)
            f = out.size()
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('Accuracy of the network on the 10000 test image: %d %%' % (100 * correct / total))
    return 100.0 * correct / total


def train():
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    #    optimizer = optim.SGD(model.parameters(), lr = LR, momentum=0.9)
    #    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    #    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    iter = 0
    num = 1
    # 训练网络
    for epoch in range(num_epoches):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            iter = iter + 1
            img, labels = data
            img, labels = img.to(device), labels.to(device)
            a = img.size()
            b = labels.size()
            optimizer.zero_grad()
            # 训练
            out = model(img)
            c = out.size()
            loss = criterion(out, labels).to(device)
            loss.backward()
            writer.add_scalar('scalar/loss', loss.item(), iter)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()  # 这一步只是学习率更新，其实应该放在epoch的循环当中

        print('epoch: %d\t batch: %d\t lr: %g\t loss: %.6f' % (
        epoch + 1, i + 1, scheduler.get_lr()[0], running_loss / (batchSize * (i + 1))))
        writer.add_scalar('scalar/train_accuracy', train_accuracy(), num + 1)
        writer.add_scalar('scalar/test_accuracy', test_accuracy(), num + 1)
        print('\n')
        num = num + 1

        torch.save(model, './model_50.pkl')


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # padding后随机裁剪
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

modelPath = './model.pth'
batchSize = 32
LR = 0.001
num_epoches = 50
writer = SummaryWriter(log_dir='scalar5')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset = torchvision.datasets.CIFAR10(root='./Cifar-10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./Cifar-10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = ResNet50(10).to(device)

if __name__ == '__main__':
    # 如果模型存在，加载模型
    # if os.path.exists(modelPath):
    #     print('model exists')
    #     model = torch.load(modelPath)
    #     print('model load')
    # else:
    #     print('model not exists')
    #     print('Training starts')
    train()
    writer.close()
    print('Training Finished')
