import math
import torch
from torch import nn
from .attention import SE
import torch.nn.functional as F


# 标准1x1卷积
class PosConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(self.bn(x))
        return x


# 标准3x3卷积
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if hasattr(self, 'bn'):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        else:
            x = self.fused_conv(x)
            x = self.relu(x)
        return x
    
    def _fuse(self):
        if not hasattr(self, 'bn'):
            return
        # 获取BN层的参数
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_gamma = self.bn.weight
        bn_beta = self.bn.bias
        bn_eps = self.bn.eps

        # 获取卷积层的权重
        conv_weight = self.conv.weight

        # 计算融合后的权重和偏置
        std = torch.sqrt(bn_var + bn_eps)
        bn_gamma = bn_gamma.view(-1, 1, 1, 1)  # [8, 1, 1, 1]
        std = std.view(-1, 1, 1, 1)  # [8, 1, 1, 1]

        # 融合后的权重
        fused_weight = conv_weight * (bn_gamma / std)

        # 融合后的偏置
        fused_bias = bn_beta - bn_mean * (bn_gamma / std).squeeze()

        # 创建新的融合卷积层
        self.fused_conv = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=True  # 融合后的卷积层需要有偏置
        )

        # 将融合后的权重和偏置复制到新的卷积层
        self.fused_conv.weight = nn.Parameter(fused_weight, requires_grad=False)
        self.fused_conv.bias = nn.Parameter(fused_bias, requires_grad=False)

        # 删除原始的卷积层和BN层
        self.__delattr__("conv")
        self.__delattr__("bn")
        return


# ResNet18的基本模块+通道注意力，效果很好
class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, 1, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
        self.c2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_c, out_c, 1)
        if out_c > 16:
            reduction = 4
        else:
            reduction = 2
        self.att = SE(out_c, reduction=reduction)

    def forward(self, x):
        res = self.res(x)
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu(res+x)
        x = self.att(x)
        return x


class MRAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.c1_hor = nn.Conv2d(in_c, out_c, (1, 3), 1, (0, 1), bias=False)
        self.c1_ver = nn.Conv2d(in_c, out_c, (3, 1), 1, (1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn1_hor = nn.BatchNorm2d(out_c)
        self.bn1_ver = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
        self.c2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.c2_hor = nn.Conv2d(out_c, out_c, (1, 3), 1, (0, 1), bias=False)
        self.c2_ver = nn.Conv2d(out_c, out_c, (3, 1), 1, (1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.bn2_hor = nn.BatchNorm2d(out_c)
        self.bn2_ver = nn.BatchNorm2d(out_c)
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_c, out_c, 1)
        if out_c > 16:
            reduction = 4
        else:
            reduction = 2
        self.att = SE(out_c, reduction=reduction)

    def forward(self, x):
        res = self.res(x)
        x = self.bn1(self.c1(x)) + self.bn1_hor(self.c1_hor(x)) + self.bn1_ver(self.c1_ver(x))
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(self.c2(x)) + self.bn2_hor(self.c2_hor(x)) + self.bn2_ver(self.c2_ver(x))
        x = self.relu(res+x)
        x = self.att(x)
        return x

# MobileNetV3的倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        mid_channels = in_channels * expansion
        self.posconv1 = PosConv(in_channels, mid_channels)
        self.midconv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.att = SE(mid_channels, reduction=4)
        self.posconv2 = PosConv(mid_channels, out_channels)
        if in_channels != out_channels:
            self.res = False
        else:
            self.res = True

    def forward(self, x):
        out = self.posconv1(x)
        out = self.midconv(out)
        out = self.att(out)
        out = self.posconv2(out)
        if self.res:
            out = x + out
        return out


# Res2Net里的基本模块
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, bias=False), nn.BatchNorm2d(planes))

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        # 对于stype=='stage‘的时候不用加上前一小块的输出结果，而是直接 sp = spx[i]
        # 是因为输入输出的尺寸不一致（通道数不一样），所以没法加起来
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        # 在这里需要加pool的原因是因为，对于每一个layer的stage模块，它的stride是不确定，layer1的stride=1
        # layer2、3、4的stride=2，前三小块都经过了stride=2的3*3卷积，而第四小块是直接送到y中的，但它必须要pool一下
        # 不然尺寸和不能和前面三个小块对应上，无法完成最后的econcat操作
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            # nn.Conv2d(dim, dim, 1, bias=False)
        )

    def forward(self, x):
        # for training/inference
        shortcut = x
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv1(x)
        return x + shortcut
    

class CSPLayerWithTwoConvCA(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True) -> None:
        super().__init__()

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, 2 * self.mid_channels, 1)
        self.final_conv = ConvModule((2 + num_blocks) * self.mid_channels, out_channels, 1)
        self.blocks = nn.ModuleList(Partial_conv3(self.mid_channels) for _ in range(num_blocks))
        self.ca = SE((2 + num_blocks) * self.mid_channels)

    def forward(self, x):
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main = torch.cat(x_main, 1)
        x_main = self.ca(x_main)
        return self.final_conv(x_main)
    

# ResNet18的基本模块+通道注意力，效果很好
class PartResidualAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=1):
        super().__init__()
        self.c1 = Partial_conv3(in_c)
        self.c2 = Partial_conv3(in_c)
        self.relu = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm2d(out_c)
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_c, out_c, 1)
        if out_c > 16:
            reduction = 4
        else:
            reduction = 2
        self.att = SE(out_c, reduction=reduction)

    def forward(self, x):
        res = self.res(x)
        x = self.c1(x)
        x = self.c2(x)
        # x = self.bn2(x)
        x = self.relu(res+x)
        x = self.att(x)
        return x


class ResidualBottleneck(nn.Module):
    """
    Residual Bottleneck Block.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution.
        expansion_factor (int): Expansion factor for the bottleneck.
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=4):
        super(ResidualBottleneck, self).__init__()
        
        # Bottleneck expansion
        bottleneck_channels = out_channels // expansion_factor
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Residual connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out