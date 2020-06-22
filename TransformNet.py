import torch.nn as nn
import torch
class TransformNet(nn.Module):
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size,self.size))
    def forward(self, input):
        out =self.net(input)
        return out/torch.sqrt(torch.sum(out**2,dim=1,keepdim=True))
class Mapping(nn.Module):
    def __init__(self, size):
        super(Mapping, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))
    def forward(self, inputs):
        outputs =self.net(inputs)
        return torch.cat((inputs,outputs),dim=-1)#/torch.sqrt(torch.sum(out**2,dim=1,keepdim=True))
class Mapping_Conv(nn.Module):
    def __init__(self,size):
        super(Mapping_Conv, self).__init__()
        self.size = size
        self.net=nn.Sequential(nn.Conv2d(512, 128, 1, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 128, 3, stride=1,padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 512, 1, 1))
    def forward(self, inputs):
        outputs=self.net(inputs)
        outputs_cat=torch.cat((inputs,outputs),dim=1)
        outputs_flatten=outputs_cat.view(outputs_cat.shape[0],-1)
        return outputs_flatten
                    
