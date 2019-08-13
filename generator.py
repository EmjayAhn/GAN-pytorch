import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dims, momentum=0.8):
        super(Generator, self).__init__()
        self.latent_dims = latent_dims
        
        self.fc1 = nn.Linear(latent_dims, 128)
        self.fc1_bn = nn.BatchNorm1d(128, momentum=momentum)
        
        self.fc2 = nn.Linear(128, 256)
        self.fc2_bn = nn.BatchNorm1d(256, momentum=momentum)
        
        self.fc3 = nn.Linear(256, 512)
        self.fc3_bn = nn.BatchNorm1d(512, momentum=momentum)
        
        self.fc4 = nn.Linear(512, 1024)
        self.fc4_bn = nn.BatchNorm1d(1024, momentum=momentum)
        
        self.fc5 = nn.Linear(1024, 784)


    def forward(self, z):
        z = self.fc1_bn(F.leaky_relu(self.fc1(z), negative_slope=0.2))
        z = self.fc2_bn(F.leaky_relu(self.fc2(z), negative_slope=0.2))
        z = self.fc3_bn(F.leaky_relu(self.fc3(z), negative_slope=0.2))
        z = self.fc4_bn(F.leaky_relu(self.fc4(z), negative_slope=0.2))
        z = self.fc5(z)
        return z



