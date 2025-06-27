# test.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
import argparse
import sys
import time
from typing import List, Tuple, Dict, Optional
import json
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from datalodaer.data_loder import BacteriaDataset
    from model.yolov11 import YOLOv11, create_model
    from utils.logger import setup_logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有必要的模块都存在")
    sys.exit(1)


class BacteriaDetectionTester:
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = 'auto'):
        """
        细菌检测测试器
        
        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径（可选）
            device: 计算设备
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 加载模型
        self.model = self._load_model()
        
        # 类别信息
        self.class_names = self.config.get('names', ['bacteria'])
        self.num_classes = len(self.class_names)
        
        # 可视化配置
        self.colors = self._generate_colors()
        
        print(f"模型加载成功: {model_path}")
        print(f"类别数: {self.num_classes}")
        print(f"类别名: {self.class_names}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        default_config = {
            'model_size': 'n',
            'num_classes': 1,
            'names': ['bacteria'],
            'img_size': 640,
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 1000
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"加载配置文件: {config_path}")
        else:
            config = {}
            print("使用默认配置")
        
        # 合并默认配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _load_model(self) -> torch.nn.Module:
        """加载训练好的模型"""
        try:
            # 尝试加载检查点
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'model' in checkpoint:
                # YOLO格式检查点
                if isinstance(checkpoint['model'], torch.nn.Module):
                    model = checkpoint['model']
                else:
                    # 可能是state_dict
                    model = create_model(
                        model_size=self.config['model_size'],
                        num_classes=self.config['num_classes']
                    )
                    model.load_state_dict(checkpoint['model'])
            elif 'model_state_dict' in checkpoint:
                # 自定义格式检查点
                model = create_model(
                    model_size=self.config['model_size'],
                    num_classes=self.config['num_classes']
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # 更新配置
                if 'config' in checkpoint:
                    self.config.update(checkpoint['config'])
            else:
                # 直接是state_dict
                model = create_model(
                    model_size=self.config['model_size'],
                    num_classes=self.config['num_classes']
                )
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e
    
    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """为每个类别生成不同的颜色"""
        colors = []
        for i in range(max(self.num_classes, 10)):
            # 使用HSV色彩空间生成颜色
            hue = i / max(self.num_classes, 10)
            color = plt.cm.Set3(hue)[:3]  # 取RGB部分
            color = tuple(int(c * 255) for c in color)
            colors.append(color)
        return colors
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """预处理图像"""
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        original_size = image.shape[:2]  # (height, width)
        
        # 调整大小
        img_size = self.config['img_size']
        image_resized = cv2.resize(image, (img_size, img_size))
        
        # 归一化
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # 转换为张量
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor, original_image, original_size
    
    def postprocess_detections(self, outputs: List[torch.Tensor], original_size: Tuple[int, int]) -> List[Dict]:
        """后处理检测结果"""
        detections = []
        img_size = self.config['img_size']
        height, width = original_size
        
        # 处理每个检测层的输出
        for i, output in enumerate(outputs):
            batch_size, channels, grid_h, grid_w = output.shape
            
            # 获取对应的步长
            stride = [8, 16, 32][i]
            
            # 创建网格
            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_h, device=output.device),
                torch.arange(grid_w, device=output.device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            
            # 重塑输出
            output = output.permute(0, 2, 3, 1)  # [batch, h, w, channels]
            
            # 分离回归和分类
            reg_max = 16
            box_output = output[..., :reg_max * 4]  # [batch, h, w, reg_max*4]
            cls_output = output[..., reg_max * 4:]   # [batch, h, w, num_classes]
            
            # DFL解码边界框
            box_output = box_output.view(batch_size, grid_h, grid_w, 4, reg_max)
            box_output = F.softmax(box_output, dim=-1)
            
            # 计算边界框坐标
            conv_weight = torch.arange(reg_max, dtype=torch.float32, device=output.device)
            box_coords = torch.sum(box_output * conv_weight.view(1, 1, 1, 1, -1), dim=-1)
            
            # 转换为中心点坐标
            box_coords[..., [0, 2]] += grid[..., 0:1]  # x坐标
            box_coords[..., [1, 3]] += grid[..., 1:2]  # y坐标
            box_coords *= stride
            
            # 转换为xywh格式
            xy_center = (box_coords[..., :2] + box_coords[..., 2:]) / 2
            wh = box_coords[..., 2:] - box_coords[..., :2]
            boxes = torch.cat([xy_center, wh], dim=-1)
            
            # 获取类别概率
            class_probs = torch.sigmoid(cls_output)
            
            # 应用置信度阈值
            conf_threshold = self.config['conf_threshold']
            max_probs, class_ids = torch.max(class_probs, dim=-1)
            
            valid_mask = max_probs > conf_threshold
            
            if valid_mask.any():
                valid_boxes = boxes[valid_mask]
                valid_probs = max_probs[valid_mask]
                valid_classes = class_ids[valid_mask]
                
                # 缩放到原始图像大小
                valid_boxes[:, [0, 2]] *= width / img_size    # x坐标
                valid_boxes[:, [1, 3]] *= height / img_size   # y坐标
                
                for box, prob, cls_id in zip(valid_boxes, valid_probs, valid_classes):
                    detections.append({
                        'bbox': box.cpu().numpy(),  # [x_center, y_center, width, height]
                        'confidence': prob.item(),
                        'class_id': cls_id.item(),
                        'class_name': self.class_names[cls_id.item()]
                    })
        
        return detections
    
    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """应用非极大值抑制"""
        if not detections:
            return []
        
        # 转换为xyxy格式进行NMS
        boxes = []
        scores = []
        
        for det in detections:
            x_center, y_center, width, height = det['bbox']
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes.append([x1, y1, x2, y2])
            scores.append(det['confidence'])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # 应用NMS
        keep_indices = torch.ops.torchvision.nms(
            boxes, scores, self.config['iou_threshold']
        )
        
        # 保留NMS后的检测结果
        nms_detections = [detections[i] for i in keep_indices]
        
        # 限制最大检测数量
        max_detections = self.config['max_detections']
        if len(nms_detections) > max_detections:
            nms_detections = sorted(nms_detections, key=lambda x: x['confidence'], reverse=True)
            nms_detections = nms_detections[:max_detections]
        
        return nms_detections
    
    def predict_single_image(self, image_path: str) -> Tuple[List[Dict], float]:
        """对单张图像进行预测"""
        start_time = time.time()
        
        # 预处理
        image_tensor, original_image, original_size = self.preprocess_image(image_path)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # 后处理
        detections = self.postprocess_detections(outputs, original_size)
        
        # NMS
        final_detections = self.apply_nms(detections)
        
        inference_time = time.time() - start_time
        
        return final_detections, inference_time
    
    def visualize_predictions(self, image_path: str, detections: List[Dict], 
                            save_path: Optional[str] = None, show_conf: bool = True) -> np.ndarray:
        """可视化预测结果"""
        # 读取原始图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建matplotlib图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # 绘制检测框
        for det in detections:
            x_center, y_center, width, height = det['bbox']
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            class_id = det['class_id']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # 获取颜色
            color = np.array(self.colors[class_id % len(self.colors)]) / 255.0
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加标签
            if show_conf:
                label = f'{class_name}: {confidence:.2f}'
            else:
                label = class_name
            
            ax.text(x1, y1 - 5, label,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                   fontsize=10, color='white', weight='bold')
        
        # 添加标题
        ax.set_title(f'检测结果 - 发现 {len(detections)} 个目标', fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            print(f"可视化结果保存至: {save_path}")
        
        # 转换为numpy数组
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return vis_image
    
    def test_dataset(self, data_yaml_path: str, output_dir: str = 'test_results') -> Dict:
        """测试整个数据集"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取数据集配置
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 获取测试集路径
        yaml_dir = Path(data_yaml_path).parent
        dataset_root = yaml_dir / Path(data_config.get('path', '.'))
        
        # 尝试使用验证集，如果没有则使用训练集
        if 'val' in data_config:
            test_images_dir = dataset_root / data_config['val']
        elif 'test' in data_config:
            test_images_dir = dataset_root / data_config['test']
        else:
            test_images_dir = dataset_root / data_config['train']
            print("警告: 未找到验证集，使用训练集进行测试")
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(test_images_dir.glob(ext)))
            image_files.extend(list(test_images_dir.glob(ext.upper())))
        
        if not image_files:
            raise ValueError(f"在 {test_images_dir} 中未找到图像文件")
        
        print(f"开始测试 {len(image_files)} 张图像...")
        
        # 统计信息
        results = {
            'total_images': len(image_files),
            'total_detections': 0,
            'avg_inference_time': 0,
            'confidence_distribution': [],
            'detections_per_image': [],
            'per_class_detections': {name: 0 for name in self.class_names}
        }
        
        total_time = 0
        
        # 创建可视化目录
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # 处理每张图像
        for i, image_path in enumerate(tqdm(image_files, desc="测试进度")):
            try:
                # 预测
                detections, inference_time = self.predict_single_image(str(image_path))
                total_time += inference_time
                
                # 更新统计
                results['total_detections'] += len(detections)
                results['detections_per_image'].append(len(detections))
                
                for det in detections:
                    results['confidence_distribution'].append(det['confidence'])
                    results['per_class_detections'][det['class_name']] += 1
                
                # 可视化前10张图像
                if i < 10:
                    vis_path = vis_dir / f"{image_path.stem}_result.jpg"
                    self.visualize_predictions(str(image_path), detections, str(vis_path))
                
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
                continue
        
        # 计算平均值
        results['avg_inference_time'] = total_time / len(image_files)
        results['avg_detections_per_image'] = np.mean(results['detections_per_image'])
        
        # 保存详细结果
        with open(output_dir / 'test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成统计图表
        self.plot_statistics(results, output_dir)
        
        return results
    
    def plot_statistics(self, results: Dict, output_dir: Path):
        """绘制统计图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 置信度分布
        if results['confidence_distribution']:
            axes[0, 0].hist(results['confidence_distribution'], bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('置信度分布')
            axes[0, 0].set_xlabel('置信度')
            axes[0, 0].set_ylabel('检测数量')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 每张图像的检测数量分布
        axes[0, 1].hist(results['detections_per_image'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('每张图像检测数量分布')
        axes[0, 1].set_xlabel('检测数量')
        axes[0, 1].set_ylabel('图像数量')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 各类别检测数量
        class_names = list(results['per_class_detections'].keys())
        class_counts = list(results['per_class_detections'].values())
        
        bars = axes[1, 0].bar(class_names, class_counts, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('各类别检测数量')
        axes[1, 0].set_xlabel('类别')
        axes[1, 0].set_ylabel('检测数量')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
        
        # 4. 总结统计
        stats_text = f"""测试总结:
        
总图像数: {results['total_images']}
总检测数: {results['total_detections']}
平均推理时间: {results['avg_inference_time']:.3f}s
平均每图检测数: {results['avg_detections_per_image']:.1f}

类别分布:
"""
        for name, count in results['per_class_detections'].items():
            percentage = (count / max(results['total_detections'], 1)) * 100
            stats_text += f"  {name}: {count} ({percentage:.1f}%)\n"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'test_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"统计图表保存至: {output_dir / 'test_statistics.png'}")
    
    def benchmark_speed(self, image_path: str, num_runs: int = 100) -> Dict:
        """性能基准测试"""
        print(f"开始性能测试 - 运行 {num_runs} 次...")
        
        # 预处理图像
        image_tensor, _, original_size = self.preprocess_image(image_path)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(image_tensor)
        
        # 同步GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 测试推理时间
        times = []
        for _ in tqdm(range(num_runs), desc="性能测试"):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # 计算统计信息
        times = np.array(times)
        results = {
            'num_runs': num_runs,
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times)),
            'device': str(self.device)
        }
        
        print(f"性能测试结果:")
        print(f"  平均推理时间: {results['mean_time']:.4f} ± {results['std_time']:.4f} 秒")
        print(f"  最快: {results['min_time']:.4f} 秒")
        print(f"  最慢: {results['max_time']:.4f} 秒")
        print(f"  平均FPS: {results['fps']:.1f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv11细菌检测模型测试')
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重文件路径 (.pt)')
    parser.add_argument('--config', type=str,
                       help='配置文件路径 (可选)')
    parser.add_argument('--source', type=str,
                       help='输入源: 图像文件路径或数据集YAML文件')
    parser.add_argument('--output', type=str, default='test_results',
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备: auto, cpu, 或 cuda')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能基准测试')
    parser.add_argument('--runs', type=int, default=100,
                       help='基准测试运行次数')
    
    args = parser.parse_args()
    
    # 创建测试器
    try:
        tester = BacteriaDetectionTester(
            model_path=args.model,
            config_path=args.config,
            device=args.device
        )
        
        # 更新配置参数
        tester.config['conf_threshold'] = args.conf
        tester.config['iou_threshold'] = args.iou
        
    except Exception as e:
        print(f"创建测试器失败: {e}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.source:
        source_path = Path(args.source)
        
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 单张图像测试
            print("=" * 50)
            print("单张图像测试")
            print("=" * 50)
            
            try:
                detections, inference_time = tester.predict_single_image(str(source_path))
                
                print(f"推理时间: {inference_time:.4f} 秒")
                print(f"检测到 {len(detections)} 个目标:")
                
                for i, det in enumerate(detections):
                    print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f}")
                
                # 可视化结果
                vis_path = output_dir / f"{source_path.stem}_result.jpg"
                tester.visualize_predictions(str(source_path), detections, str(vis_path))
                
                # 保存检测结果
                results = {
                    'image_path': str(source_path),
                    'inference_time': inference_time,
                    'detections': detections
                }
                
                with open(output_dir / 'detection_results.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"结果保存至: {output_dir}")
                
            except Exception as e:
                print(f"测试失败: {e}")
                import traceback
                traceback.print_exc()
        
        elif source_path.suffix.lower() == '.yaml':
            # 数据集测试
            print("=" * 50)
            print("数据集测试")
            print("=" * 50)
            
            try:
                results = tester.test_dataset(str(source_path), str(output_dir))
                
                print(f"\n测试完成!")
                print(f"总图像数: {results['total_images']}")
                print(f"总检测数: {results['total_detections']}")
                print(f"平均推理时间: {results['avg_inference_time']:.4f} 秒")
                print(f"平均每图检测数: {results['avg_detections_per_image']:.1f}")
                
                print(f"\n类别统计:")
                for name, count in results['per_class_detections'].items():
                    percentage = (count / max(results['total_detections'], 1)) * 100
                    print(f"  {name}: {count} ({percentage:.1f}%)")
                
                print(f"\n详细结果保存至: {output_dir}")
                
            except Exception as e:
                print(f"数据集测试失败: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print(f"不支持的文件格式: {source_path.suffix}")
    
    # 性能基准测试
    if args.benchmark:
        print("=" * 50)
        print("性能基准测试")
        print("=" * 50)
        
        if args.source and Path(args.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            benchmark_image = args.source
        else:
            # 创建一个测试图像
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            benchmark_image = output_dir / 'benchmark_test.jpg'
            cv2.imwrite(str(benchmark_image), test_img)
            print(f"使用生成的测试图像: {benchmark_image}")
        
        try:
            benchmark_results = tester.benchmark_speed(str(benchmark_image), args.runs)
            
            # 保存基准测试结果
            with open(output_dir / 'benchmark_results.json', 'w', encoding='utf-8') as f:
                json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"性能测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()