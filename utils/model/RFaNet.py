import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from .NLRF_add_3x3conv_before_D7 import NLRF_block
from .DFA import DFA_block

__all__ = ['RFaNet']

defaultcfg = {
    9 : ['DFA', 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'NLRF', 'L'],
}

num_classes=25

class RFaNet(nn.Module):
    def __init__(self, dataset='mydataset', depth=9, init_weights=True, cfg=None):
        super(RFaNet, self).__init__()

        if cfg is None:
            cfg = defaultcfg[depth]
            print(cfg)
        self.cfg = cfg            
        self.feature = self.make_layers(cfg, True)

        if dataset == 'mydataset':
            global num_classes
            num_classes = 25

        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1############change############
        for v in cfg:
            if v == 'DFA':
                layers += [DFA_block(in_channels,64)]
                in_channels = 64
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'NLRF':
                layers += [NLRF_block(in_channels,in_channels)]
            elif v == 'L':
                conv2d = nn.Conv2d(in_channels, num_classes, kernel_size=1, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(num_classes), nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    # layers += [conv2d, nn.BatchNorm2d(v), nn.Tanh()]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                    # layers += [conv2d, nn.Tanh()]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):  # output = model(data) --> Start work

        x = self.feature(x)

        x = nn.AdaptiveAvgPool2d(1)(x) #########important#######

        x = x.view(x.size(0), -1)  # 完成数据类型的转换

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5) # Gamma ---I guess
                m.bias.data.zero_() # Beta ---I guess
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                #m.bias.data.zero_()

if __name__ == '__main__':
    net = RFaNet()
    print(net)
    x = Variable(torch.FloatTensor(16, 1, 64, 64))############change############
    y = net(x)
    print(y.data.shape)
    print(y)
