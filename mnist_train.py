from functools import total_ordering
from numpy import argmax
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from torchvision import transforms


#Basic Params-----------------------------
epoch = 1
learning_rate = 0.01
batch_size_train = 64
batch_size_test = 1000
gpu = torch.cuda.is_available()
momentum = 0.5

#Load Data-------------------------------
train_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
                                transform=torchvision.transforms.Compose([                  
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([  
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)

train_data_size = len(train_loader)
test_data_size = len(test_loader)

#Define Model----------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=128),
            nn.Linear(in_features=128, out_features=10),
        )
    
    def forward(self, x):
        return self.model(x)

if gpu:
    net = Net().cuda()
else:
    net = Net()


#Define Loss and Optimizer----------------

if gpu: 
    loss_fn = nn.CrossEntropyLoss().cuda()
else:
    loss_fn = nn.CrossEntropyLoss()


optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


#Define Tensorboard-------------------

writer = SummaryWriter(log_dir='logs/{}'.format(time.strftime('%Y%m%d-%H%M%S')))

#Train---------------------------------

total_train_step = 0

def train(epoch):
    global total_train_step
    total_train_step = 0
    for data in train_loader:
        imgs,targets = data
        if gpu:
            imgs,targets = imgs.cuda(),targets.cuda()
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = loss_fn(outputs,targets)
        loss.backward()
        optimizer.step()
        if total_train_step % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, total_train_step, train_data_size,
                100. * total_train_step / train_data_size, loss.item()))
        writer.add_scalar('loss', loss.item(), total_train_step)
        total_train_step += 1


#Test---------------------------------

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            if gpu:
                imgs,targets = imgs.cuda(),targets.cuda()
            outputs = net(imgs)
            _,predicted = torch.max(outputs.data,1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print('Test Accuracy: {}/{} ({:.0f}%)'.format(correct, total, 100.*correct/total))
    return correct/total


#Run----------------------------------
 

for i in range(1,epoch+1):
    print("-----------------Epoch: {}-----------------".format(i))
    train(i)
    test()
    writer.add_scalar('test_accuracy', test(), total_train_step)
    #save model
    torch.save(net,'model/mnist_model.pth')
    print('Saved model')

writer.close()

 
#Evaluate---------------------------------

model = torch.load("./model/mnist_model.pth")
model.eval()
print(model)

fig = plt.figure(figsize=(20,20))
for i in range(20):
    data = test_loader.dataset[i]
    if gpu:
        img = data[0].cuda()
    else:
        img = data[0]
    img = torch.reshape(img,(1,1,28,28))
    with torch.no_grad():
        output = model(img)
    #plot the image and the predicted number
    fig.add_subplot(4,5,i+1)
    plt.title(argmax(output.data.cpu().numpy()))
    plt.imshow(data[0].numpy().squeeze(),cmap='gray')
plt.show()