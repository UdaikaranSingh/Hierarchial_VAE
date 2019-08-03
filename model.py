import torch as torch
import torch.nn as nn
import torch.nn.functional as F

#reference: https://arxiv.org/pdf/1804.02086.pdf?fbclid=IwAR2T3fO5ZK1ivc0wEHP_WhMiI1vyrGVxUl3Pt8rErD3mLcLLcwMuBxi8MQc
#page: 13


class Encoder(nn.Module):
    
    def __init__(self, batch_size):
        super(Encoder, self).__init__()
        
        
        #storing variables:
        self.batch_size = batch_size;
        
        #architecture:
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= 4, stride= 2)
        self.bn1 = nn.BatchNorm2d(num_features= 32)
        
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 4, stride= 2)
        self.bn2 = nn.BatchNorm2d(num_features= 32)
        
        self.conv3 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 4, stride= 2)
        self.bn3 = nn.BatchNorm2d(num_features= 64)
        
        self.conv4 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 4, stride= 2)
        self.bn4 = nn.BatchNorm2d(num_features= 64)
        
        self.fully_connected1 = nn.Linear(in_features= 256, out_features= 256)
        self.fully_connected2 = nn.Linear(in_features= 256, out_features= 256)
        self.fully_connected3 = nn.Linear(in_features= 256, out_features= 20)
        self.fully_connected4 = nn.Linear(in_features= 20, out_features= 20)
        
    
    def forward(self, images):
        #images are of shape (batch size x 3 x 64 x 64 )
        out = F.relu(self.bn1(self.conv1(images)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        
        #flatten vector
        out = out.view((self.batch_size, 256))
        
        #fully connected layers
        out = F.relu(self.fully_connected1(out))
        out = F.relu(self.fully_connected2(out))
        out = F.relu(self.fully_connected3(out))
        out = F.relu(self.fully_connected4(out))
        
        return out
    
        

class Decoder(nn.Module):
    
    def __init__(self, batch_size):
        super(Decoder, self).__init__()
        
        #storing variables
        self.batch_size = batch_size
        
        #architectures
        self.fully_connected1 = nn.Linear(in_features= 20, out_features= 256)
        self.fully_connected2 = nn.Linear(in_features= 256, out_features= 1024)
        
        self.upconv1 = nn.ConvTranspose2d(in_channels= 64, out_channels= 64, kernel_size= 3, stride= 2)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.upconv2 = nn.ConvTranspose2d(in_channels= 64, out_channels= 32, kernel_size= 5, stride= 3)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.upconv3 = nn.ConvTranspose2d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.upconv4 = nn.ConvTranspose2d(in_channels= 32, out_channels= 3, kernel_size= 4, stride= 2)

    def forward(self, z):
        
        #z is a vector of length (batch size x 20).
        out = F.relu(self.fully_connected1(z))
        out = F.tanh(self.fully_connected2(out))
        
        #reshape
        out = out.view((self.batch_size, 64, 4, 4))
        
        #upconvolution
        out = F.relu(self.bn1(self.upconv1(out)))
        print(out.shape)
        out = F.relu(self.bn2(self.upconv2(out)))
        print(out.shape)
        out = F.relu(self.bn3(self.upconv3(out)))
        print(out.shape)
        out = self.upconv4(out)
        
        return out
        
        