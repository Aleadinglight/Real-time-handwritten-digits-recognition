import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import pickle
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input channels -> Output channels, kernel = kernel_size x kernel_size 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        # Choose randomly some cells and set them to zero the prevent overfitting
        # Also, to reduce the size of the network 
        # By default it is 0.5
        # When we forward, we have to multiply every cell with 0.5 (propability theory)
        # There is option depends on we forward and backprop in Pytorch, find self.training
        self.conv2_drop = nn.Dropout2d()
        # This is the fully connected layer
        # The first dimension is Channel_size x Width x Height
        self.fc1 = nn.Linear(320,50)
        # The output is 26 since there is 26 symbol in the alphabet 
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        # Max_pool is choosing each 2x2 blocks and return max 
        # ( pool size here is 2, and the stride is by default = pool size = 2)
        # This means that the dimension of the picture will be reduces in half
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # -1 in view meaning that we not sure about the dimension, but the other one we know
        # We know that the other one is 320
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def inp(epoch):

    random.shuffle(data)
    # Input must be Batch size x Num channel x Height x Width
    # If input is a tensor then do not use np.shape() to check size, use .size() method

    Num_channels = 1
    Batch_size = 32
    Height, Width = data[0][0].shape
    n = len(data)

    for i in range(0, n, Batch_size):
        input = []
        target = []
        # Create a batch, then turn it to Tensor and feed it to the neuralnetwork
        for k in range(0,Batch_size):
            input.append(data[k][0])
            target.append(data[k][1])

        # if convert from numpy input = torch.from_numpy(input)
        input = torch.Tensor(input)
        # When call torch_numpy it returns a double Tensors, we need Float Tensors
        # input = input.float()
        
        # Put a new dimension in
        input.unsqueeze_(0)       
        input = input.view(Batch_size,Num_channels,Height,Width)
        target = Variable(torch.LongTensor(target)).cuda()
        input = Variable(input).cuda()
        train(i, input, target)
        
    test(epoch)

def test(epoch):
    model.eval()
    Num_channels = 1
    Height, Width = test_data[0][0].shape
    n = len(test_data)
    input = []
    target = []
    for k in range(0, n):
        input.append(test_data[k][0])
        target.append(test_data[k][1])

    input = torch.Tensor(input)
    input.unsqueeze_(0)       
    input = input.view(-1,Num_channels,Height,Width)
    target = Variable(torch.LongTensor(target)).cuda()
    input = Variable(input).cuda()

    out = model(input)
    _, pred = out.max(1)
    d = 0
    for i in range(0,n):
        if pred[i].data[0] == target[i].data[0]:
            d+=1
    accuracy = float(1.0*d/n)*100
    print "Epoch number",epoch,": accuracy = ",accuracy,"%"
    
    if (ans[-1]<=accuracy):
        ans.append(accuracy)
        torch.save(model.state_dict(), save_file)
        print "Saved model"

def train(batch, input, target): 

    model.train()
    for i in range(0,10): 
        out = model(input)
        loss = F.nll_loss(out, target)
        #_, pred = out.max(1)
        #print loss.data[0]
        #if loss.data[0]-0<= 1e-3:
        #    _, pred = out.max(1)
        #    break
        opt.zero_grad()
        loss.backward()
        opt.step()


model = Net()
opt = optim.Adam(model.parameters(), lr = 0.0001)
if torch.cuda.is_available():
    model.cuda()

file_name = "mnist_data.dat"
with open(file_name,"rb") as file:
    data = pickle.load(file)

file_name = "mnist_test.dat"
with open(file_name,"rb") as file:
    test_data = pickle.load(file)

save_file = "mnist_parameters.inp"
ans = [0]
try:
    model.load_state_dict(torch.load(save_file))
    print "load saved data"
except IOError:
    torch.save(model.state_dict(), save_file)
    print "init"
    model.load_state_dict(torch.load(save_file))

for epoch in range(40):
    inp(epoch)

