## utils/metrics.py
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

class DetectionMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.total_predictions = 0
        self.total_ground_truth = 0
        
    def update(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """
        更新指标
        
        Args:
            pred_boxes: 预测框 [N, 4] (x1, y1, x2, y2)
            gt_boxes: 真实框 [M, 4] (x1, y1, x2, y2)
            iou_threshold: IoU阈值
        """
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            return
        
        if len(pred_boxes) == 0:
            self.fn += len(gt_boxes)
            self.total_ground_truth += len(gt_boxes)
            return
            
        if len(gt_boxes) == 0:
            self.fp += len(pred_boxes)
            self.total_predictions += len(pred_boxes)
            return
        
        # 计算IoU矩阵
        iou_matrix = self.calculate_iou_matrix(pred_boxes, gt_boxes)
        
        # 匹配预测框和真实框
        matched_gt = set()
        matched_pred = set()
        
        # 按IoU排序进行匹配
        iou_flat = iou_matrix.flatten()
        indices = np.unravel_index(np.argsort(iou_flat)[::-1], iou_matrix.shape)
        
        for pred_idx, gt_idx in zip(indices[0], indices[1]):
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
                
            if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
                self.tp += 1
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
        
        # 计算FP和FN
        self.fp += len(pred_boxes) - len(matched_pred)
        self.fn += len(gt_boxes) - len(matched_gt)
        
        self.total_predictions += len(pred_boxes)
        self.total_ground_truth += len(gt_boxes)
    
    def calculate_iou_matrix(self, boxes1, boxes2):
        """计算IoU矩阵"""
        # 扩展维度以便广播
        boxes1 = np.expand_dims(boxes1, axis=1)  # [N, 1, 4]
        boxes2 = np.expand_dims(boxes2, axis=0)  # [1, M, 4]
        
        # 计算交集
        x1 = np.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
        y1 = np.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
        x2 = np.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
        y2 = np.minimum(boxes1[:, :, 3], boxes2[:, :, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算并集
        area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
        area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
        union = area1 + area2 - intersection
        
        # 计算IoU
        iou = intersection / (union + 1e-6)
        return iou
    
    def get_metrics(self):
        """获取指标"""
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn
        }
    
    def print_metrics(self):
        """打印指标"""
        metrics = self.get_metrics()
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    metrics = DetectionMetrics()
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            
            # 模型预测
            predictions = model(images)
            
            # 处理每张图片的预测结果
            for pred, target in zip(predictions, targets):
                pred_boxes = pred['boxes'].cpu().numpy()
                gt_boxes = target['boxes'].cpu().numpy()
                
                metrics.update(pred_boxes, gt_boxes)
    
    return metrics.get_metrics()