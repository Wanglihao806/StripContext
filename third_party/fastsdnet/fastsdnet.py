import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ResidualAttentionBlock, GolLocAttention, ConvModule
from .common import shuffle_chnls


# MobileCrackMerge指的是带上了稠密融合模块，MobieCrackMergeEdge指稠密模块和Edge模块
# 没有稠密模块大概51~52mIoU，加上了有55.
class DenselyShuffleMerge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            GolLocAttention(out_channels, out_channels//2),
        )

    def forward(self, x):
        x = shuffle_chnls(x, 5)
        x = self.conv(x)
        return x


class MobileEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        r = [16, 32, 32]
        self.downsam = nn.Sequential(ConvModule(in_channels, r[0], stride=2), ConvModule(r[0], r[0], stride=2))
        self.e1 = ResidualAttentionBlock(r[0], r[0])
        self.e2 = ResidualAttentionBlock(r[0], r[1])
        self.e3 = ResidualAttentionBlock(r[1], r[2])
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.downsam(x)
        e1 = self.e1(x)
        e2 = self.e2(self.down(e1))
        e3 = self.e3(self.down(e2))
        return e1, e2, e3


class MobileDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        r = [16, 32, 32]
        self.merge1 = DenselyShuffleMerge(r[0] + r[1] + r[2], r[0])
        self.merge2 = DenselyShuffleMerge(r[0] + r[1] + r[2], r[1])
        self.merge3 = DenselyShuffleMerge(r[0] + r[1] + r[2], r[2])

        self.d3 = nn.Conv2d(r[2], num_classes, 1)
        self.d2 = nn.Conv2d(r[1], num_classes, 1)
        self.d1 = nn.Conv2d(r[0], num_classes, 1)

        self.final = nn.Conv2d(num_classes * 3, num_classes, 1)

    def forward(self, e1, e2, e3):
        h1, w1 = e1.shape[-2:]
        h2, w2 = e2.shape[-2:]
        h3, w3 = e3.shape[-2:]

        # 完成不同尺度的变换
        s1 = torch.cat((e1, F.interpolate(e2, (h1, w1), mode='bilinear', align_corners=True), F.interpolate(e3, (h1, w1), mode='bilinear', align_corners=True)), dim=1)
        s2 = F.interpolate(s1, (h2, w2), mode='bilinear', align_corners=True)
        s3 = F.interpolate(s1, (h3, w3), mode='bilinear', align_corners=True)
        m1 = self.merge1(s1)
        m2 = self.merge2(s2)
        m3 = self.merge3(s3)
        d3 = F.interpolate(self.d3(m3), (h1, w1), mode='bilinear', align_corners=True)
        d2 = F.interpolate(self.d2(m2), (h1, w1), mode='bilinear', align_corners=True)
        d1 = self.d1(m1)

        final = self.final(torch.cat([d3, d2, d1], dim=1))
        return final, d1, d2, d3


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.down = nn.MaxPool2d(2, 2)
        self.layer1 = ResidualAttentionBlock(32, 64)
        self.seg = nn.Conv2d(64, num_classes, kernel_size=1)
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.down(x)
        x = self.layer1(x)
        x = self.down(x)
        x_seg = self.seg(x)
        x = self.aap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        if self.training:
            return x, x_seg
        else:
            return x


class FastSDNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.feature = MobileEncoder(in_channels)
        self.seg = MobileDecoder(num_classes)

    def forward(self, x):
        size = x.shape[-2:]
        e1, e2, e3 = self.feature(x)
        out = list(self.seg(e1, e2, e3))
        for i in range(len(out)):
            out[i] = F.interpolate(out[i], size=size, mode='bilinear', align_corners=True)
        return out

    def export_onnx(self, out_path):
        import os
        class FeatureModel(nn.Module):
            def __init__(self, feature):
                super(FeatureModel, self).__init__()
                self.feature = feature

            def forward(self, x):
                e1, e2, e3 = self.feature(x)
                score = self.cls(e3)
                return e1, e2, e3, score

        # feature_cls_model = FeatureModel(self.feature, self.cls)
        dummy_input = torch.rand(1, 3, 512, 256)
        feature_cls_axes = {'input': {0: 'batch', 2: 'height', 3: 'weight'},
                            'e1': {0: 'batch', 2: 'height', 3: 'weight'},
                            'e2': {0: 'batch', 2: 'height', 3: 'weight'},
                            'e3': {0: 'batch', 2: 'height', 3: 'weight'},
                            'score': {0: 'batch'}}
        seg_axes = {'e1': {0: 'batch', 2: 'height', 3: 'weight'},
                    'e2': {0: 'batch', 2: 'height', 3: 'weight'},
                    'e3': {0: 'batch', 2: 'height', 3: 'weight'},
                    'output': {0: 'batch', 2: 'height', 3: 'weight'}}
        # feature_cls_trace = torch.jit.trace(feature_cls_model, dummy_input)
        # torch.onnx.export(feature_cls_trace, dummy_input, os.path.join(out_path, 'feature_cls.onnx'), input_names=['input'],
        #                   output_names=['e1', 'e2', 'e3', 'score'], dynamic_axes=feature_cls_axes)
        e1, e2, e3 = self.feature(dummy_input)
        inputs = (torch.rand_like(e1), torch.rand_like(e2), torch.rand_like(e3))
        seg_trace = torch.jit.trace(self.seg, inputs)
        torch.onnx.export(seg_trace, inputs, os.path.join(out_path, 'seg_model.onnx'), input_names=['e1', 'e2', 'e3'],
                          output_names=['output'], dynamic_axes=seg_axes)
        return



def export_onnx(model, out_path, dummy_input):
    model.eval()
    model_trace = torch.jit.trace(model, dummy_input)

    dynamic_axes_0 = {
        'input': {0: 'batch', 2: 'height', 3: 'weight'},
        'output': {0: 'batch'},
    }
    #os.makedirs(out_path, exist_ok=True)
    torch.onnx.export(model_trace, dummy_input, f'{out_path}_{model._get_name()}.onnx',
                      input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes_0)
    print(f'out_path: {out_path}_{model._get_name()}.onnx' )



if __name__ == "__main__":
    import numpy as np
    b = 1
    t = torch.randn(1, 3, 320, 400)

    # t = torch.rand(b, 3, 320, 400)
    model = FastSDNet(3, 1)
    model.eval()
    # model.export_onnx('../..')
    s = model(t)
    print(s)
    model.parameters()

