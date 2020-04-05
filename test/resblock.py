import torch
import torch.nn as nn
from math import sqrt

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=True)

class conv_bn_relu_res_block(nn.Module):
    def __init__(self):
        super(conv_bn_relu_res_block, self).__init__()
        self.conv1 = conv3x3(64, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.mul(out,0.1)
        out = torch.add(out,residual) 
        return out    
    
class resblock(nn.Module):
    def __init__(self, block, block_num, input_channel, output_channel):
        super(resblock, self).__init__()

        self.in_channels = input_channel
        self.out_channels = output_channel
        self.input_conv = conv3x3(self.in_channels, out_channels=64)  
        self.conv_seq = self.make_layer(block, block_num)
        self.conv = conv3x3(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.output_conv = conv3x3(in_channels=64,  out_channels=self.out_channels)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,sqrt(2./n))
                
    def make_layer(self,block,num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block()) 
        return nn.Sequential(*layers)   
    
    def forward(self, x):
       
        out = self.input_conv(x)
        residual = out
        out = self.conv_seq(out)
        out = self.conv(out)
        out = torch.add(out,residual)  
        out = self.relu(out)
        out = self.output_conv(out)
        return out
