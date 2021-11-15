import torch

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import copy

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

batch_size = 256
epoch = 30

transforms_base = transforms.Compose([transforms.Resize((64,64)),
                                      transforms.ToTensor()])

train_dataset = ImageFolder(root='./splitted/train',
                            transform=transforms_base)
val_dataset = ImageFolder(root='./splitted/val',
                          transform=transforms_base)
test_dataset = ImageFolder(root='./splitted/test',
                          transform=transforms_base)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4) #num_workers  data 로딩을 위해 몇 개의 서브 프로세스를 사용할 것인지를 결정 https://jybaek.tistory.com/799
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4)

class NET(nn.Module):
    def __init__(self):
        super(NET,self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 33)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2

        return F.log_softmax(x)

model_base = NET().to(device)
optimizer = optim.Adam(model_base.parameters(), lr=0.001)

def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_gred()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output,
                                         target,
                                         reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return  test_loss, test_accuracy

def train_baseline(model, train_loader, val_loader,
                   optimizer, num_epochs = 30):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range (1, num_epochs + 1):
        since = time.time()
        train(model, train_loader, optimizer)
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since

        print('----------epoch {}-----------'.format(epoch))
        print('train_loss : {:.4f}, accuracy : {:.2f}'.format(train_loss, train_acc))
        print('completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed %60))
    model.load_state_dict(best_model_wts)
    return model

base = train_baseline(model_base, train_loader, val_loader, optimizer, epoch)

torch.save(base,'./model/baseline.pt')









