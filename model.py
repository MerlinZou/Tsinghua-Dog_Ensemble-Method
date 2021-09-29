from jittor.models import densenet161
from jittor.models import googlenet
from jittor.models import resnet50
from jittor.models import vgg19_bn
from jittor.models import resnext101_32x8d
import jittor.nn as nn 

class Net1(nn.Module):
    def __init__(self, num_classes):
        self.base_net = densenet161(num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        return x 

class Net2(nn.Module):
    def __init__(self, num_classes):
        self.base_net = googlenet(num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        return x 

class Net3(nn.Module):
    def __init__(self, num_classes):
        self.base_net = resnet50(num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        return x 

class Net4(nn.Module):
    def __init__(self, num_classes):
        self.base_net = vgg19_bn(num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        return x 

class Net5(nn.Module):
    def __init__(self, num_classes):
        self.base_net = resnext101_32x8d(num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        return x 
