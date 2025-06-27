#model/yolov11.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import yaml
from pathlib import Path

def make_divisible(x, divisor=8):
    """确保通道数可以被divisor整除"""
    return max(divisor, int(x + divisor / 2) // divisor * divisor)

class Conv(nn.Module):
    """标准卷积块"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def autopad(self, k, p):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

class C2f(nn.Module):
    """C2f模块 - YOLOv11的核心模块"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):
    """标准瓶颈层"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SPPF(nn.Module):
    """空间金字塔池化快速版本"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Detect(nn.Module):
    """检测头"""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x

class DFL(nn.Module):
    """分布焦点损失"""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class YOLOv11(nn.Module):
    """YOLOv11模型"""
    def __init__(self, nc=80, model_size='n'):
        super().__init__()
        self.nc = nc
        self.model_size = model_size
        
        # 模型尺寸配置
        self.model_configs = {
            'n': {'depth': 0.33, 'width': 0.25, 'channels': [64, 128, 256, 512, 1024]},
            's': {'depth': 0.33, 'width': 0.50, 'channels': [64, 128, 256, 512, 1024]},
            'm': {'depth': 0.67, 'width': 0.75, 'channels': [96, 192, 384, 768, 1536]},
            'l': {'depth': 1.00, 'width': 1.00, 'channels': [128, 256, 512, 1024, 2048]},
            'x': {'depth': 1.00, 'width': 1.25, 'channels': [160, 320, 640, 1280, 2560]}
        }
        
        config = self.model_configs[model_size]
        depth_multiple = config['depth']
        width_multiple = config['width']
        channels = config['channels']
        
        # 计算实际通道数
        ch = [make_divisible(c * width_multiple) for c in channels]
        
        # Backbone
        self.backbone = nn.Sequential(
            Conv(3, ch[0], 3, 2),  # P1/2
            Conv(ch[0], ch[1], 3, 2),  # P2/4
            C2f(ch[1], ch[1], max(round(3 * depth_multiple), 1), True),
            Conv(ch[1], ch[2], 3, 2),  # P3/8
            C2f(ch[2], ch[2], max(round(6 * depth_multiple), 1), True),
            Conv(ch[2], ch[3], 3, 2),  # P4/16
            C2f(ch[3], ch[3], max(round(6 * depth_multiple), 1), True),
            Conv(ch[3], ch[4], 3, 2),  # P5/32
            C2f(ch[4], ch[4], max(round(3 * depth_multiple), 1), True),
            SPPF(ch[4], ch[4], 5),
        )
        
        # 保存特征通道数
        self.p3_ch = ch[2]  # P3 output channels
        self.p4_ch = ch[3]  # P4 output channels  
        self.p5_ch = ch[4]  # P5 output channels
        
        # Neck - FPN
        self.neck = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2f(ch[4] + ch[3], ch[3], max(round(3 * depth_multiple), 1)),
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2f(ch[3] + ch[2], ch[2], max(round(3 * depth_multiple), 1)),
            Conv(ch[2], ch[2], 3, 2),
            C2f(ch[2] + ch[3], ch[3], max(round(3 * depth_multiple), 1)),
            Conv(ch[3], ch[3], 3, 2),
            C2f(ch[3] + ch[4], ch[4], max(round(3 * depth_multiple), 1)),
        ])
        
        # Head
        self.head = Detect(nc, (ch[2], ch[3], ch[4]))
        
        self.stride = torch.tensor([8., 16., 32.])
        self.head.stride = self.stride
        
    def forward(self, x):
        # Backbone features
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 9]:  # P3, P4, P5
                features.append(x)
        
        p3, p4, p5 = features
        
        # Neck - FPN
        # Top-down pathway
        x = self.neck[0](p5)  # Upsample P5
        x = torch.cat([x, p4], 1)  # Concat with P4
        x = self.neck[1](x)  # C2f
        p4_out = x
        
        x = self.neck[2](x)  # Upsample
        x = torch.cat([x, p3], 1)  # Concat with P3
        x = self.neck[3](x)  # C2f
        p3_out = x
        
        # Bottom-up pathway
        x = self.neck[4](p3_out)  # Conv
        x = torch.cat([x, p4_out], 1)  # Concat
        x = self.neck[5](x)  # C2f
        p4_final = x
        
        x = self.neck[6](x)  # Conv
        x = torch.cat([x, p5], 1)  # Concat
        x = self.neck[7](x)  # C2f
        p5_final = x
        
        # Head
        return self.head([p3_out, p4_final, p5_final])
    
    def load_pretrained_weights(self, weights_path):
        """加载预训练权重"""
        if weights_path and Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 过滤不匹配的键
            model_dict = self.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
            print(f"加载预训练权重: {weights_path}")
            print(f"匹配的参数: {len(filtered_dict)}/{len(state_dict)}")

def create_model(model_size='n', num_classes=80, pretrained_weights=None):
    """创建YOLO模型"""
    model = YOLOv11(nc=num_classes, model_size=model_size)
    
    if pretrained_weights:
        model.load_pretrained_weights(pretrained_weights)
    
    return model