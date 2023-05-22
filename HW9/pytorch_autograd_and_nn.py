import torch
import torch.nn as nn
from a4_helper import *
import torch.nn.functional as F
import torch.optim as optim

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from pytorch_autograd_and_nn.py!')
  

class PlainBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.net = None

    if not downsample: 
      self.net = nn.Sequential(OrderedDict([
      ('BN1', nn.BatchNorm2d(Cin)),
      ('relu1', nn.ReLU()),
      ('conv1', nn.Conv2d(Cin, Cout, 3, padding=1)),
      ('BN2', nn.BatchNorm2d(Cout)),
      ('relu2', nn.ReLU()),
      ('conv2', nn.Conv2d(Cout, Cout, 3, padding=1)),
      ]))

    else:
      self.net = nn.Sequential(OrderedDict([
      ('BN1', nn.BatchNorm2d(Cin)),
      ('relu1', nn.ReLU()),
      ('conv1', nn.Conv2d(Cin, Cout, 3, padding=1)),
      ('BN2', nn.BatchNorm2d(Cout)),
      ('relu2', nn.ReLU()),
      ('conv2', nn.Conv2d(Cout, Cout, 3, padding=1, stride =2)),
      ]))

  def forward(self, x):
    return self.net(x)
    

class ResNet(nn.Module):
  def __init__(self, stage_args, Cin=3, block=PlainBlock, num_classes=100):
    super().__init__()

    self.cnn = None
    net = [ResNetStem(Cin=Cin, Cout=stage_args[0][0])]
    if block == block:
      net.extend([ResNetStage(*arg, block) for arg in stage_args])

    self.cnn = nn.Sequential(*net)

    self.fc = nn.Linear(stage_args[-1][1], num_classes)
  
  def forward(self, x):
    scores = None
    
    x = self.cnn(x)
    x = F.avg_pool2d(x, kernel_size=x.shape[-1])
    scores = self.fc(x.view(x.shape[0], -1))
    
    return scores
    

class ResNetStem(nn.Module):
  def __init__(self, Cin=3, Cout=8):
    super().__init__()
    layers = [
        nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
    ]
    self.net = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.net(x)

class ResNetStage(nn.Module):
  def __init__(self, Cin, Cout, num_blocks, downsample=True,
               block=PlainBlock):
    super().__init__()
    blocks = [block(Cin, Cout, downsample)]
    for _ in range(num_blocks - 1):
      blocks.append(block(Cout, Cout))
    self.net = nn.Sequential(*blocks)
  
  def forward(self, x):
    return self.net(x)
    
    