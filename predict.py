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

def predict(input): 

    model = Net()
    model.eval()
    save_file = "mnist_parameters_94.23.inp"
    try:
        model.load_state_dict(torch.load(save_file))
        #print "Load saved data."
    except IOError:
        print "No parameters."
    
    Num_channels = 1
    Batch_size = 1
    Height, Width = input.shape
    input = input/255
    input = torch.from_numpy(input)
    input.unsqueeze_(0)  
    input.unsqueeze_(0)    
    input = input.float()
    input = Variable(input) 
    #print input.size()
    #input = input.view(Batch_size,Num_channels,Height,Width)
    out = model(input)
    _, pred = out.max(1)
    print "Predicted value: ",pred.data[0][0]

if __name__ == "__main__":
    print "main"
    #test(1)