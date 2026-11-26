import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from third_party.fastsdnet.fastsdnet import FastSDNet
from third_party.SPNet.models import resnet as SP
from third_party.CoordAttention import coordatt as CA

def up(x,size):return F.interpolate(x,size=size,mode='bilinear',align_corners=True)

class StripContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce=nn.Conv2d(32,16,1,bias=False)
        self.local=nn.Conv2d(16,16,3,padding=1,groups=16,bias=False)
        self.row=nn.Conv2d(16,16,(3,1),padding=(1,0),groups=16,bias=False)
        self.column=nn.Conv2d(16,16,(1,3),padding=(0,1),groups=16,bias=False)
        self.norm=nn.BatchNorm2d(16)
        self.project=nn.Conv2d(16,32,1,bias=False)
        nn.init.zeros_(self.project.weight)
    def forward(self,e3):
        z=self.reduce(e3)
        # Pool only one spatial axis; preserve location along the other axis.
        row=self.row(z.mean(dim=3,keepdim=True))
        column=self.column(z.mean(dim=2,keepdim=True))
        context=self.local(z)+row+column
        return e3+self.project(F.relu(self.norm(context)))

class BranchSubset(StripContext):
    def __init__(self,branches):
        super().__init__()
        for name in ('local','row','column'):
            if name not in branches:delattr(self,name)
    def forward(self,e3):
        z=self.reduce(e3)
        local=self.local(z) if hasattr(self,'local') else torch.zeros_like(z)
        row=self.row(z.mean(dim=3,keepdim=True)) if hasattr(self,'row') else 0
        column=self.column(z.mean(dim=2,keepdim=True)) if hasattr(self,'column') else 0
        context=local+row+column
        return e3+self.project(torch.nn.functional.relu(self.norm(context)))

class AxisMean(nn.Module):
    """Fixed-shape equivalent of AdaptiveAvgPool2d((None,1)/(1,None))."""
    def __init__(self,dim):super().__init__();self.dim=dim
    def forward(self,x):return x.mean(dim=self.dim,keepdim=True)

def coordinate_attention():
    m=CA.CoordAtt(32,32,reduction=32)
    m.pool_h=AxisMean(3);m.pool_w=AxisMean(2)
    return m

class SPM(nn.Module):
    def __init__(self):
        super().__init__();self.gate=SP.SPBlock(32,32,norm_layer=nn.BatchNorm2d)
        self.gate.pool1=AxisMean(3);self.gate.pool2=AxisMean(2)
        # Match the author's enclosing ResNet initialization (bias retains default).
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                m.weight.data.normal_(0,math.sqrt(2./(m.kernel_size[0]*m.kernel_size[1]*m.out_channels)))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1);m.bias.data.zero_()
    def forward(self,x):return x*self.gate(x)

BRANCHES = {
    "FSD_StripLocal": ("local",),
    "FSD_StripAxes": ("row", "column"),
    "FSD_StripLocalRow": ("local", "row"),
    "FSD_StripLocalColumn": ("local", "column"),
}
VARIANTS = ("FastSDNet", "FSD_StripContext", "FSD_E3SPM", "FSD_E3CoordAtt",
            *BRANCHES, "FSD_StripAtE1", "FSD_StripAtE2", "FSD_StripAtE2E3")


class DecoderContext(nn.Module):
    """Transform selected decoder inputs after computing all encoder outputs."""

    def __init__(self, base, name, seed):
        super().__init__()
        self.feature, self.seg = base.feature, base.seg
        self.index = {"FSD_StripAtE1": 0, "FSD_StripAtE2": 1}.get(name, 2)
        self.joint = name == "FSD_StripAtE2E3"
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed + 7100009)
            if name in BRANCHES:
                block = BranchSubset(BRANCHES[name])
            elif name == "FSD_E3SPM":
                block = SPM()
            elif name == "FSD_E3CoordAtt":
                block = coordinate_attention()
            else:
                block = StripContext()
            if self.index == 0:
                block.reduce = nn.Conv2d(16, 16, 1, bias=False)
                block.project = nn.Conv2d(16, 16, 1, bias=False)
                nn.init.zeros_(block.project.weight)
            if self.joint:
                self.e3_context = block
                self.e2_context = copy.deepcopy(block)
            else:
                self.context = block

    def forward(self, x):
        features = list(self.feature(x))
        if self.joint:
            features[1] = self.e2_context(features[1])
            features[2] = self.e3_context(features[2])
        else:
            features[self.index] = self.context(features[self.index])
        return [up(head, x.shape[-2:]) for head in self.seg(*features)]


def build_model(name, seed):
    if name not in VARIANTS:
        raise ValueError(f"Unknown configuration: {name}")
    torch.manual_seed(seed)
    backbone = FastSDNet(3, 2)
    return backbone if name == "FastSDNet" else DecoderContext(backbone, name, seed)

