import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class domain_cls(nn.Module):

    def __init__(self,in_channel=256,num_domains = 4):
        super(domain_cls, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2)
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_domains)
        self.softmax = nn.Softmax(dim=1)
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, reverse_feature):
        x = F.relu(self.conv1(reverse_feature))
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        domain_output = torch.flatten(x, 1)
        domain_output = self.fc(domain_output)
        domain_output = self.softmax(domain_output)
        return domain_output

class jig_cls(nn.Module):

    def __init__(self,in_channel=256,jig_classes = 31):
        super(jig_cls, self).__init__()

        self.pool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channel, jig_classes)
        self.softmax = nn.Softmax(dim=1)
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.pool3(x)
        domain_output = torch.flatten(x, 1)
        domain_output = self.fc(domain_output)
        domain_output = self.softmax(domain_output)
        return domain_output



class img_domain_cls(nn.Module):

    def __init__(self,in_channel=3):
        super(img_domain_cls, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, 7)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, reverse_feature):
        x = F.relu(self.conv1(reverse_feature))
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        domain_output = torch.flatten(x, 1)
        domain_output = self.fc(domain_output)
        domain_output = self.softmax(domain_output)
        return domain_output



# bottleneck
class residueBlock(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3, stride=1, padding=1):
        super(residueBlock, self).__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if inplane == outplane:
            self.dimension_inc = False
        else:
            self.dimension_inc = True
        self.conv1 = nn.Conv2d(self.inplane, self.outplane//2, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.outplane//2)

        self.conv2 = nn.Conv2d(self.outplane//2, self.outplane, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm2d(self.outplane)
        self.projection_shortcut = nn.Conv2d(self.inplane, self.outplane, 1, stride=self.stride, bias=False)
        return

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        if self.dimension_inc == True:
            shortcut = self.projection_shortcut(shortcut)

        x = x + shortcut
        x = F.relu(x)
        return x



class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DomainDisentanglementBlock(nn.Module):
    def __init__(self, channel, reduction=16,num_domains=6):
        super(DomainDisentanglementBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
                    nn.Sigmoid()
        )
        self.d = domain_cls_simple(channel)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x_res = F.relu(x - y)
        x_pool = self.avg_pool(x_res)
        d_pred_adv = self.d(x_pool)
        d_pred = self.d(y)
        return x_res,d_pred_adv,d_pred

class domain_cls_simple(nn.Module):
    def __init__(self,in_channel=256,num_domains=6):
        super(domain_cls_simple, self).__init__()

        self.fc = nn.Linear(in_channel, num_domains)
        self.softmax = nn.Softmax(dim=1)
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        domain_output = torch.flatten(x, 1)
        domain_output = self.fc(domain_output)
        domain_output = self.softmax(domain_output)
        return domain_output