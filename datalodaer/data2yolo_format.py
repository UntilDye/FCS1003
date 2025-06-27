#dataloader/data2yolo_format.py
import os
import json
import shutil
from PIL import Image
import random
from pathlib import Path

class DatasetConverter:
    def __init__(self, image_dir, json_dir, output_dir, train_ratio=0.8):
        """
        初始化数据集转换器
        
        Args:
            image_dir: 原始图片目录
            json_dir: JSON标注文件目录
            output_dir: 输出YOLO格式目录
            train_ratio: 训练集比例
        """
        self.image_dir = Path(image_dir)
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        
        # 创建输出目录结构
        self.create_directories()
        
        # 类别映射 - 根据你的数据集调整
        self.class_mapping = {
            'colony': 0,  # 菌落类别
            'bacteria': 0,  # 如果有其他名称
        }
    
    def create_directories(self):
        """创建YOLO格式的目录结构"""
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def convert_json_to_yolo(self, json_data, img_width, img_height):
        """
        将JSON标注转换为YOLO格式
        
        Args:
            json_data: JSON标注数据
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            list: YOLO格式的标注列表
        """
        yolo_annotations = []
        
        labels = json_data.get('labels', [])
        for label in labels:
            class_name = label.get('class', 'colony')
            
            # 获取类别ID
            class_id = self.class_mapping.get(class_name.lower(), 0)
            
            # 获取边界框坐标
            x = label.get('x', 0)
            y = label.get('y', 0)
            width = label.get('width', 0)
            height = label.get('height', 0)
            
            # 转换为YOLO格式 (中心点坐标 + 归一化)
            center_x = (x + width / 2) / img_width
            center_y = (y + height / 2) / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            
            # 确保坐标在[0,1]范围内
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        return yolo_annotations
    
    def process_dataset(self):
        """处理整个数据集"""
        # 获取所有图片文件
        image_files = list(self.image_dir.glob('*.jpg')) + \
                     list(self.image_dir.glob('*.jpeg')) + \
                     list(self.image_dir.glob('*.png'))
        
        if not image_files:
            raise ValueError(f"在 {self.image_dir} 中未找到图片文件")
        
        # 随机打乱并分割数据集
        random.shuffle(image_files)
        train_count = int(len(image_files) * self.train_ratio)
        train_files = image_files[:train_count]
        val_files = image_files[train_count:]
        
        print(f"总计: {len(image_files)} 张图片")
        print(f"训练集: {len(train_files)} 张图片")
        print(f"验证集: {len(val_files)} 张图片")
        
        # 处理训练集
        self.process_split(train_files, 'train')
        
        # 处理验证集
        self.process_split(val_files, 'val')
        
        # 创建数据集配置文件
        self.create_dataset_yaml()
        
        print("数据集转换完成!")
    
    def process_split(self, image_files, split):
        """处理特定分割的数据"""
        successful_conversions = 0
        
        for img_file in image_files:
            try:
                # 获取对应的JSON文件
                json_file = self.json_dir / f"{img_file.stem}.json"
                
                if not json_file.exists():
                    print(f"警告: 未找到 {img_file.name} 对应的JSON文件")
                    continue
                
                # 读取图片获取尺寸
                with Image.open(img_file) as img:
                    img_width, img_height = img.size
                
                # 读取JSON标注
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 转换为YOLO格式
                yolo_annotations = self.convert_json_to_yolo(json_data, img_width, img_height)
                
                # 复制图片到目标目录
                target_img_path = self.output_dir / 'images' / split / img_file.name
                shutil.copy2(img_file, target_img_path)
                
                # 保存YOLO格式标注
                target_label_path = self.output_dir / 'labels' / split / f"{img_file.stem}.txt"
                with open(target_label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                successful_conversions += 1
                
            except Exception as e:
                print(f"处理 {img_file.name} 时出错: {e}")
        
        print(f"{split} 集成功转换: {successful_conversions} 个文件")
    
    def create_dataset_yaml(self):
        """创建YOLO数据集配置文件"""
        yaml_content = f"""
path: {self.output_dir.absolute()}
train: images/train
val: images/val

nc: 1  # 类别数量
names: ['colony']  # 类别名称
"""
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content.strip())
        
        print(f"数据集配置文件已保存: {yaml_path}")

def main():
    # 配置路径
    IMAGE_DIR = r"C:\code\bateria\dataset\Images"
    JSON_DIR = r"C:\code\bateria\dataset\DatasetJson"
    OUTPUT_DIR = r"C:\code\bateria\dataset\yolo_format"
    
    # 创建转换器并处理数据集
    converter = DatasetConverter(IMAGE_DIR, JSON_DIR, OUTPUT_DIR, train_ratio=0.8)
    converter.process_dataset()

if __name__ == "__main__":
    main()