import torch
from torch import nn
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F
from torch import Tensor

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x

class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())

class CPC(nn.Module):
    def __init__(self, input_c: int, output_c: int, expand_ratio: int, drop_rate: float):
        super(CPC, self).__init__()
        self.expand_channels = expand_ratio*input_c


        self.Conv1 = nn.Conv2d(in_channels = input_c,
                               out_channels = self.expand_channels,
                               kernel_size=(1,1),
                               stride=(1,1),
                               padding=(0,0),
                               bias=False)

        self.Norm1 = nn.BatchNorm2d(self.expand_channels)
        self.Ac1 = nn.SiLU()

        self.Pool = nn.AvgPool2d(kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.Norm2 = nn.BatchNorm2d(self.expand_channels)
        self.Ac2 = nn.SiLU()
        self.Conv2 = nn.Conv2d(in_channels = self.expand_channels,
                               out_channels = output_c,
                               kernel_size=(1,1),
                               stride=(1,1),
                               padding=(0,0),
                               bias=False)
        self.Norm3 = nn.BatchNorm2d(output_c)
        self.Ac3 = nn.SiLU()

    def forward(self,x):

        x1 = self.Conv1(x)
        x2 = self.Norm1(x1)
        x3 = self.Ac1(x2)
        x4 = self.Pool(x3)
        x5 = self.Norm2(x4)
        x6 = self.Ac2(x5) + x3
        x7 = self.Conv2(x6)
        x8 = self.Norm3(x7)
        x9 = self.Ac3(x8) + x
        return x9

class LM(nn.Module):
    def __init__(self, dim:int,  expand_ratio: int,repeats:int,drop_rate = 0):
        super(LM, self).__init__()
        blocks = OrderedDict()
        for i in range(repeats):
            blocks.update({str(i): CPC(dim,dim,expand_ratio,drop_rate)})
        self.blocks = nn.Sequential(blocks)
    def forward(self,x):
        result = self.blocks(x)
        return result

class FusedMBConv(nn.Module):
    def __init__(self, input_c:int, output_c:int, stride:int):
        super(FusedMBConv, self).__init__()

        self.shortcut = stride == 1

        self.expanded_conv = ConvBNActivation(in_planes=input_c,
                                              out_planes=input_c*4,
                                              kernel_size= 3,
                                              stride= stride)
        self.SeConv = SqueezeExcitation(input_c = input_c*4)
        self.project_conv = ConvBNActivation(in_planes=input_c*4,
                                            out_planes= output_c,
                                            kernel_size = 1,
                                            stride = 1)

    def forward(self, x):
        result = self.expanded_conv(x)
        result1 = self.SeConv(result)
        result2 = self.project_conv(result1)

        if self.shortcut:
            result2 += x

        return result2

class MetaMo(nn.Module):
    def __init__(self,ratio=3,num_classes =160):
        super(MetaMo, self).__init__()
        dims = [64,128,196,270]
        repeats = [2,2,4,2]
        self.conv0 = ConvBNActivation(in_planes=3, out_planes=32, kernel_size=3, stride=2)

        self.conv1 = FusedMBConv(input_c=32, output_c=64, stride=2)
        self.conv2 = FusedMBConv(input_c=64, output_c=64, stride=1)

        self.M1 = LM(dims[0],ratio,repeats[0]) #56
        self.Conv3 = ConvBNActivation(in_planes=dims[0],out_planes=dims[1],kernel_size=3,stride=2)
        self.M2 = LM(dims[1], ratio, repeats[1])#28
        self.Conv4 = ConvBNActivation(in_planes=dims[1], out_planes=dims[2], kernel_size=3, stride=2)
        self.M3 = LM(dims[2], ratio, repeats[2])#14
        self.Conv5 = ConvBNActivation(in_planes=dims[2], out_planes=dims[3], kernel_size=3, stride=2)
        self.M4 = LM(dims[3], ratio, repeats[3])#7
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dims[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        y1 = self.M1(self.conv2(self.conv1(self.conv0(x))))
        y2 = self.Conv3(y1)
        y3 = self.M2(y2)
        y4 = self.Conv4(y3)
        y5 = self.M3(y4)
        y6 = self.Conv5(y5)
        y7 = self.M4(y6)

        y8 =  torch.flatten(self.avgpool(y7),1)

        y9 = self.classifier(y8)
        return y9


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

if __name__ == '__main__':

    a = torch.randn(4,3,224,224)

    model = MetaMo()
    print(model)
    b = model(a)
    model_structure(model)
    print(b.size())
    torch.save(model.state_dict(), ".\\meta14.pth")