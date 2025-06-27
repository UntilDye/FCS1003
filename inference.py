# inference.py
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys

class BacteriaDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        细菌检测器
        
        Args:
            model_path: 模型权重路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """
        加载模型，处理不同的权重格式
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            # 首先尝试直接加载为 YOLO 模型
            model = YOLO(str(model_path))
            print(f"成功加载 YOLO 模型: {model_path}")
            return model
            
        except KeyError as e:
            if "'model'" in str(e):
                print(f"检测到自定义训练的权重格式，正在转换...")
                return self._load_custom_weights(model_path)
            else:
                raise e
                
    def _load_custom_weights(self, model_path):
        """
        加载自定义训练的权重
        """
        try:
            # 加载检查点
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 获取配置信息
            config = checkpoint.get('config', {})
            model_size = config.get('model_size', 'n')
            num_classes = config.get('num_classes', 1)
            
            print(f"检测到配置: 模型大小={model_size}, 类别数={num_classes}")
            
            # 创建基础 YOLO 模型
            if model_size == 'n':
                base_model = YOLO('yolo11n.pt')
            elif model_size == 's':
                base_model = YOLO('yolo11s.pt')
            elif model_size == 'm':
                base_model = YOLO('yolo11m.pt')
            elif model_size == 'l':
                base_model = YOLO('yolo11l.pt')
            elif model_size == 'x':
                base_model = YOLO('yolo11x.pt')
            else:
                base_model = YOLO('yolo11n.pt')
                
            # 如果类别数不是80（COCO），需要修改模型
            if num_classes != 80:
                # 获取模型配置
                model_yaml = base_model.model.yaml
                model_yaml['nc'] = num_classes  # 设置类别数
                
                # 重新创建模型
                from ultralytics.nn.tasks import DetectionModel
                model = DetectionModel(model_yaml)
                
                # 加载预训练权重到backbone和neck
                try:
                    model_state = checkpoint['model_state_dict']
                    
                    # 创建新的状态字典，过滤掉不匹配的层
                    new_state_dict = {}
                    model_dict = model.state_dict()
                    
                    for name, param in model_state.items():
                        if name in model_dict and param.shape == model_dict[name].shape:
                            new_state_dict[name] = param
                        else:
                            print(f"跳过不匹配的层: {name}")
                    
                    # 加载状态字典
                    model.load_state_dict(new_state_dict, strict=False)
                    print(f"成功加载 {len(new_state_dict)} 个匹配的层")
                    
                except Exception as e:
                    print(f"加载权重时出错: {e}")
                    print("使用预训练模型")
                
                # 创建 YOLO 包装器
                yolo_model = YOLO(model=model)
                return yolo_model
                
            else:
                # 标准的80类模型，直接加载权重
                model_state = checkpoint['model_state_dict']
                base_model.model.load_state_dict(model_state, strict=False)
                return base_model
                
        except Exception as e:
            print(f"加载自定义权重失败: {e}")
            print("尝试使用预训练模型")
            return YOLO('yolo11n.pt')
    
    def predict(self, image_path, save_dir=None):
        """
        对单张图片进行预测
        """
        try:
            results = self.model(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=save_dir is not None,
                project=save_dir if save_dir else None
            )
            return results
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def predict_batch(self, image_dir, save_dir=None):
        """
        对目录中的所有图片进行批量预测
        """
        image_dir = Path(image_dir)
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(ext)))
            image_files.extend(list(image_dir.glob(ext.upper())))
        
        if not image_files:
            print(f"在目录 {image_dir} 中未找到图片文件")
            return []
        
        print(f"找到 {len(image_files)} 张图片")
        
        results = []
        for img_file in image_files:
            try:
                result = self.predict(str(img_file), save_dir)
                if result:
                    results.append(result)
                    
                    # 打印检测结果
                    if len(result[0].boxes) > 0:
                        colony_count = len(result[0].boxes)
                        print(f"{img_file.name}: 检测到 {colony_count} 个菌落")
                    else:
                        print(f"{img_file.name}: 未检测到菌落")
                else:
                    print(f"{img_file.name}: 预测失败")
                    
            except Exception as e:
                print(f"处理 {img_file.name} 时出错: {e}")
                continue
        
        return results
    
    def visualize_results(self, image_path, results, save_path=None):
        """
        可视化检测结果
        """
        if not results or len(results) == 0:
            print("没有检测结果可视化")
            return
            
        # 读取原始图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 绘制检测框
        if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制置信度
                label = f'Colony: {conf:.2f}'
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示结果
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title(f'检测结果: {len(results[0].boxes) if hasattr(results[0], "boxes") else 0} 个菌落')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        plt.show()
    
    def count_colonies(self, image_path):
        """
        统计菌落数量
        """
        results = self.predict(image_path)
        if results and hasattr(results[0], 'boxes'):
            count = len(results[0].boxes)
        else:
            count = 0
        return count

def main():
    parser = argparse.ArgumentParser(description='细菌检测推理')
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--source', type=str, required=True,
                       help='输入图片或目录路径')
    parser.add_argument('--output', type=str, default='runs/detect/predict',
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU阈值')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--save-vis', type=str,
                       help='可视化结果保存路径')
    
    args = parser.parse_args()
    
    try:
        # 创建检测器
        print(f"正在加载模型: {args.model}")
        detector = BacteriaDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        # 创建输出目录
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查输入是文件还是目录
        source_path = Path(args.source)
        
        if source_path.is_file():
            # 单张图片预测
            print(f"处理单张图片: {source_path}")
            results = detector.predict(str(source_path), args.output)
            
            if results:
                colony_count = detector.count_colonies(str(source_path))
                print(f"检测到 {colony_count} 个菌落")
                
                if args.visualize:
                    save_path = args.save_vis if args.save_vis else str(output_dir / f"{source_path.stem}_result.jpg")
                    detector.visualize_results(str(source_path), results, save_path)
            else:
                print("预测失败")
                
        elif source_path.is_dir():
            # 批量预测
            print(f"处理目录: {source_path}")
            results = detector.predict_batch(str(source_path), args.output)
            print(f"完成 {len(results)} 张图片的检测")
            
            # 统计总数
            total_colonies = 0
            for result in results:
                if result and hasattr(result[0], 'boxes'):
                    total_colonies += len(result[0].boxes)
            
            print(f"总共检测到 {total_colonies} 个菌落")
            
        else:
            print(f"错误: 无效的输入路径 {source_path}")
            sys.exit(1)
            
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    #python inference.py --model runs\detect\bacteria_detection\best.pt --source dataset\yolo_format\images\val --output runs\test --conf 0.5 --iou 0.3 --visualize