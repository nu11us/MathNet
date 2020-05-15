import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import default_rng
from torch.utils.tensorboard import SummaryWriter
import sys
from os import path

BATCH_SIZE = 512
NUM_WORKERS = 6

LEARNING_RATE = 0.01
WEIGHT_DECAY =  0.0
MOMENTUM = 0.98
LR_STEP = 10

LENGTH = 100000
torch.backends.cudnn.deterministic = True

class MathList:
    def __init__(self, count, mode):
        self.count = count
        self.mode = mode
    def generate(self):
        lst = []
        rng = default_rng()
        for i in range(self.count):
            matrix = np.random.randn(1,2)
            if mode == 'add':
                val = matrix[0][0] + matrix[0][1]
            elif mode == 'subtract':
                val = matrix[0][0] - matrix[0][1]
            elif mode == 'multiply':
                val = matrix[0][0] * matrix[0][1]
            else:
                val = matrix[0][0] / matrix[0][1]
            lst.append([matrix,val])
        return lst

class MathDataset(torch.utils.data.Dataset):
    def __init__(self, length, mode = 'add'):
        super(MathDataset).__init__()
        self.data = MathList(count = length, mode = mode).generate()
    def __getitem__(self, index):
        return (self.data[index][1], torch.from_numpy(self.data[index][0]))
    def __len__(self):
        return len(self.data)

class MathNet(nn.Module):
    def __init__(self, mode=1):
        super(MathNet, self).__init__()        
        
        if mode == 0:
            self.fc1 = nn.Linear(2,2)
            self.fc2 = nn.Linear(2,1)
            self.layers = [self.fc1, self.fc2]
        elif mode == 1:
            self.fc1 = nn.Linear(2, 8)
            self.fc2 = nn.Linear(8, 1)
            self.layers = [self.fc1, self.fc2] 
        elif mode == 2:
            self.fc1 = nn.Linear(2, 8)
            self.fc2 = nn.Linear(8, 32)
            self.fc3 = nn.Linear(32, 1)
            self.layers = [self.fc1, self.fc2, self.fc3]
        elif mode == 3:
            self.fc1 = nn.Linear(2, 8)
            self.fc2 = nn.Linear(8, 32)
            self.fc3 = nn.Linear(32, 64)
            self.fc4 = nn.Linear(64, 1)
            self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self,x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
def train(trainloader, epochs):
    for epoch in epochs:
        run_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs, labels = data[1].to(device, dtype=torch.float), data[0].to(device, dtype=torch.float) 
            outputs = model(inputs)
            labels = labels.unsqueeze(1).unsqueeze(1)
            loss_val = criterion(outputs, labels)
            loss_val.backward()
            optimizer.step()   
            
            run_loss += loss_val.item()
            correct += (abs(outputs-labels) < 0.001).sum(dim=0).item()
            total += len(outputs)

        writer.add_scalar('Loss/train', run_loss, epoch)
        writer.add_scalar('Accuracy/train', correct/total, epoch)
        run_loss = 0.0
        correct = 0.0
        total = 0.0

def test(testloader, number=0):
    run_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i,data in enumerate(testloader, 0):
            inputs, labels = data[1].to(device, dtype=torch.float), data[0].to(device, dtype=torch.float) 
            outputs = model(inputs)
            labels = labels.unsqueeze(1).unsqueeze(1)
            loss_val = criterion(outputs, labels)
            
            run_loss += loss_val.item()
            correct += (abs(outputs-labels) < 0.001).sum(dim=0).item()
            total += len(outputs)
            
    writer.add_scalar('Loss/test', run_loss, number)
    writer.add_scalar('Accuracy/test', correct/total, number) 

def run(trainloader, testloader, num_epochs):
    for i in range(5, num_epochs, 5):
        train(trainloader,range(i-5,i))
        test(testloader,number=i)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        mode = str(sys.argv[1])
        vers = int(sys.argv[2])
    elif len(sys.argv) > 1:
        mode = str(sys.argv[1])
        vers = 0 
    else:
        mode = 'add'
        vers = 0
    
    writer = SummaryWriter()
    data = MathDataset(length = LENGTH, mode = mode)
    train_data, test_data = torch.utils.data.random_split(data, [int(.75*len(data)), int(.25*len(data))])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)

    model = MathNet(mode=vers)
    criterion = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion.to(device)
    
    optimizer = torch.optim.SGD([{'params':model.parameters()}, {'params': criterion.parameters()}],lr=LEARNING_RATE)
    
    i = 1
    name = "{}{}.pth".format(mode,i)
    
    while path.exists(name):
        i += 1
        name = "{}{}.pth".format(mode,i)
    
    run(train_loader, test_loader, 1000) 
        
    torch.save(model.state_dict(), name)
