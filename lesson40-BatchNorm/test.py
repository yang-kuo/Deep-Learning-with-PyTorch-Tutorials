import torch 
import torch.nn as nn

input = torch.arange(0, 100).view(-1, 4, 5, 5).float()
print(input)
# bn1 = nn.BatchNorm1d(20)
bn1 = nn.BatchNorm2d(4)
output = bn1(input)

print(output.shape)