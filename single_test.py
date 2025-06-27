# quick_test.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from test import BacteriaDetectionTester


def quick_test_image(model_path: str, image_path: str, output_path: str = None):
    """快速测试单张图像"""
    try:
        # 创建测试器
        tester = BacteriaDetectionTester(model_path)
        
        # 预测
        detections, inference_time = tester.predict_single_image(image_path)
        
        # 打印结果
        print(f"推理时间: {inference_time:.4f} 秒")
        print(f"检测到 {len(detections)} 个目标:")
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f} "
                  f"位置: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
        
        # 可视化
        if output_path is None:
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_result.jpg"
        
        tester.visualize_predictions(image_path, detections, str(output_path))
        
        return detections, inference_time
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='快速测试单张图像')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--image', type=str, required=True, help='图像路径')
    parser.add_argument('--output', type=str, help='输出路径')
    
    args = parser.parse_args()
    
    quick_test_image(args.model, args.image, args.output)