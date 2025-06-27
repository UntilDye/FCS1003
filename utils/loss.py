# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算IoU, GIoU, DIoU, CIoU
    
    Args:
        box1: 形状为 (N, 4) 的张量
        box2: 形状为 (M, 4) 的张量  
        xywh: 如果为True，输入格式为(x_center, y_center, width, height)
        GIoU, DIoU, CIoU: 选择IoU变体
    
    Returns:
        IoU值张量，形状为 (N, M)
    """
    # 获取边界框坐标
    if xywh:  # 转换 xywh 到 xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # xyxy格式
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 计算交集区域
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # 计算并集
    union = w1 * h1 + w2 * h2 - inter + eps

    # 计算IoU
    iou = inter / union
    
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 最小外接矩形宽度
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 最小外接矩形高度
        
        if CIoU or DIoU:  # 距离IoU
            c2 = cw ** 2 + ch ** 2 + eps  # 对角线距离平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + 
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 中心点距离平方
            
            if CIoU:  # 完全IoU
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            
            return iou - rho2 / c2  # DIoU
        
        # GIoU
        c_area = cw * ch + eps  # 最小外接矩形面积
        return iou - (c_area - union) / c_area  # GIoU
    
    return iou  # 标准IoU


def smooth_BCE(eps=0.1):
    """标签平滑的二元交叉熵"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    """Focal Loss实现"""
    
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class YOLOv11Loss(nn.Module):
    """YOLOv11损失函数"""
    
    def __init__(self, model, autobalance=False):
        super().__init__()
        device = next(model.parameters()).device
        
        h = model.head  # Detect层
        
        # 损失权重
        self.cp, self.cn = smooth_BCE(eps=0.0)  # 正负样本权重
        
        # Focal loss
        g = 0.0  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(nn.BCEWithLogitsLoss(), gamma=g), FocalLoss(nn.BCEWithLogitsLoss(), gamma=g)
        else:
            BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

        # 类别标签平滑
        self.cp, self.cn = smooth_BCE(eps=0.0)
        
        # 损失函数
        self.BCEcls, self.BCEobj = BCEcls, BCEobj
        
        # 检测层参数
        self.balance = [4.0, 1.0, 0.4]  # P3, P4, P5的权重平衡
        self.ssi = list(h.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, {}, autobalance
        
        # 损失权重
        self.box = 7.5
        self.cls = 0.5
        self.obj = 1.0
        
        self.na = h.nl  # 检测层数量
        self.nc = h.nc  # 类别数量
        self.nl = h.nl  # 检测层数量
        self.anchors = h.stride  # [8, 16, 32]
        self.device = device

    def __call__(self, predictions, targets):
        """
        计算损失
        
        Args:
            predictions: 模型输出 [P3, P4, P5]，每个元素形状为 [batch_size, anchors*(nc+5), height, width]
            targets: 真实标签 [N, 6] (batch_idx, class, x, y, w, h)
            
        Returns:
            总损失
        """
        device = targets.device
        lcls = torch.zeros(1, device=device)  # 分类损失
        lbox = torch.zeros(1, device=device)  # 回归损失
        lobj = torch.zeros(1, device=device)  # 目标性损失
        
        # 构建目标
        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)
        
        # 计算每个检测层的损失
        for i, pi in enumerate(predictions):  # 每个检测层
            b, a, gj, gi = indices[i]  # 图像索引，anchor索引，网格y，网格x
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)  # 目标obj
            
            n = b.shape[0]  # 目标数量
            if n:
                # 预测子集对应目标
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                
                # 回归损失
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # 预测框
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # IoU
                lbox += (1.0 - iou).mean()  # IoU损失
                
                # 目标性损失
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # IoU比值
                
                # 分类损失
                if self.nc > 1:  # 只有在多类别时才计算分类损失
                    t = torch.full_like(pcls, self.cn, device=device)  # 目标
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE
                    
            # 目标性损失
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
            
        lbox *= self.box
        lobj *= self.obj
        lcls *= self.cls
        bs = tobj.shape[0]  # batch size
        
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """构建训练目标"""
        na, nt = 1, targets.shape[0]  # anchor数量，目标数量
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # 标准化到网格空间的增益
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # 相同的anchor索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # 追加anchor索引

        g = 0.5  # 偏移
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=targets.device).float() * g  # 偏移

        for i in range(self.nl):
            anchors, shape = torch.tensor([8, 16, 32], device=targets.device)[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # 将目标匹配到anchor
            t = targets * gain  # 形状(3,n,7)
            if nt:
                # 匹配
                r = t[..., 4:6] / anchors  # wh比率
                j = torch.max(r, 1 / r).max(2)[0] < 4  # 比较 # 4是超参数
                t = t[j]  # 过滤

                # 偏移
                gxy = t[:, 2:4]  # 网格xy
                gxi = gain[[2, 3]] - gxy  # 逆向
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # 定义
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # 网格索引

            # 追加
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(torch.tensor([8, 16, 32], device=targets.device)[i])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class SimplifiedLoss(nn.Module):
    """简化的YOLO损失函数 - 用于快速启动和测试"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        """
        简化损失计算
        
        Args:
            predictions: 模型输出 [P3, P4, P5]
            targets: 真实标签 [N, 6] (batch_idx, class, x, y, w, h)
        """
        
        # 如果没有目标，返回零损失
        if targets.size(0) == 0:
            device = predictions[0].device if isinstance(predictions, list) else predictions.device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 简单的目标函数 - 仅用于确保训练流程正常
        # 在实际使用中，你需要实现完整的YOLO损失函数
        total_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        
        if isinstance(predictions, list):
            for pred in predictions:
                # 简单的L2损失
                loss = torch.mean(pred ** 2) * 0.01
                total_loss = total_loss + loss
        else:
            total_loss = torch.mean(predictions ** 2) * 0.01
            
        return total_loss


def create_loss_function(model, loss_type='yolov11'):
    """
    创建损失函数
    
    Args:
        model: YOLO模型
        loss_type: 损失函数类型 ('yolov11' 或 'simplified')
    
    Returns:
        损失函数实例
    """
    if loss_type == 'yolov11':
        return YOLOv11Loss(model)
    elif loss_type == 'simplified':
        return SimplifiedLoss()
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")


# 用于测试的工具函数
def test_loss_function():
    """测试损失函数"""
    import sys
    sys.path.append('.')
    from model.yolov11 import create_model
    
    # 创建模型
    model = create_model(model_size='n', num_classes=1)
    model.eval()
    
    # 创建测试数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 640, 640)
    
    # 创建测试目标 (batch_idx, class, x, y, w, h)
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.3, 0.3],  # 第一张图片的目标
        [1, 0, 0.3, 0.4, 0.2, 0.4],  # 第二张图片的目标
    ])
    
    # 前向传播
    predictions = model(images)
    
    # 测试简化损失
    simple_loss = SimplifiedLoss()
    loss_simple = simple_loss(predictions, targets)
    print(f"简化损失: {loss_simple.item():.6f}")
    
    # 测试完整损失
    try:
        yolo_loss = YOLOv11Loss(model)
        loss_yolo, loss_items = yolo_loss(predictions, targets)
        print(f"YOLOv11损失: {loss_yolo.item():.6f}")
        print(f"损失分量 (box, obj, cls): {loss_items.tolist()}")
    except Exception as e:
        print(f"YOLOv11损失计算出错: {e}")
        print("使用简化损失函数继续训练")


if __name__ == "__main__":
    test_loss_function()