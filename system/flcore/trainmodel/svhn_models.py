import torch.nn.functional as F
import torch


class CNN2(torch.nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(8*8*16, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x


class CNN3(torch.nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(4*4*32, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x


class CNN3b(torch.nn.Module):
    def __init__(self):
        super(CNN3b, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 20, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(40)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(4*4*40, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x) # 16 16
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x) # 
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x) # 4 4
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x) # 
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x


class CNN3c(torch.nn.Module):
    def __init__(self):
        super(CNN3c, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(4*4*20, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x


class CNN4(torch.nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(2*2*64, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x


class CNN4b(torch.nn.Module):
    def __init__(self):
        super(CNN4b, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 20, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(40)
        self.conv3 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(80)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(2*2*80, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x
    
 
class CNN4c(torch.nn.Module):
    def __init__(self):
        super(CNN4c, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(40)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(2*2*40, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x


class CNN5(torch.nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*128, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x


class CNN5b(torch.nn.Module):
    def __init__(self):
        super(CNN5b, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 20, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(40)
        self.conv3 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(80)
        self.conv4 = torch.nn.Conv2d(80, 100, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(100)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*100, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x


class CNN5c(torch.nn.Module):
    def __init__(self):
        super(CNN5c, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1) # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*128, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn0(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x