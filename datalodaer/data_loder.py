# dataloader/data_loader.py 
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from typing import List, Tuple, Dict, Any


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    自定义批次处理函数，用于处理不同数量边界框的样本。
    将图片和标签从批次中分离，并为标签添加正确的批次索引。
    """
    images, targets = zip(*batch)
    
    # 将图片堆叠成一个批次张量
    images = torch.stack(images, 0)
    
    # 为每个样本的标签添加批次索引
    targets_with_batch_idx = []
    for i, target in enumerate(targets):
        # target[:, 0] 是批次索引列
        target[:, 0] = i 
        targets_with_batch_idx.append(target)
        
    # 将所有标签拼接在一起
    targets = torch.cat(targets_with_batch_idx, 0)
    
    return images, targets


class BacteriaDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, img_size: int = 640, augment: bool = False, class_names: List[str] = None):
        """
        细菌检测数据集

        Args:
            img_dir: 图片目录路径
            label_dir: 标签目录路径
            img_size: 图片尺寸
            augment: 是否使用数据增强
            class_names: 类别名称列表
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.class_names = class_names or ['bacteria']
        
        # 获取所有图片文件
        self.img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.img_files.extend(list(self.img_dir.glob(ext)))
            self.img_files.extend(list(self.img_dir.glob(ext.upper())))
            
        print(f"在 {self.img_dir} 中找到 {len(self.img_files)} 张图片")
        
        # 定义数据增强
        if augment:

            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(1, 3)),
                    A.MotionBlur(blur_limit=3),
                ], p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 读取图片
        img_path = self.img_files[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法读取图片: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    values = line.strip().split()
                    if len(values) >= 5:
                        class_id = int(values[0])
                        x_center, y_center, width, height = map(float, values[1:5])
                        
                        # 验证标注是否有效
                        if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < width <= 1 and 0 < height <= 1):
                            boxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
        
        # 应用数据增强
        try:
            if self.transform:
                transformed = self.transform(
                    image=image,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                image = transformed['image']
                boxes = transformed['bboxes']
                class_labels = transformed['class_labels']
        except Exception as e:
            print(f"数据增强失败 {img_path}: {e}")
            # 如果增强失败，使用基础变换，确保流程不中断
            basic_transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            transformed = basic_transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            
        # 转换为目标格式
        num_boxes = len(boxes)

        # 这里的第一个元素 (batch_idx) 暂时用0填充，它将在 collate_fn 中被正确赋值
        targets = torch.zeros((num_boxes, 6)) # [batch_idx, class_id, x, y, w, h]
        
        if num_boxes > 0:
            targets[:, 1] = torch.tensor(class_labels)    # class
            targets[:, 2:] = torch.tensor(boxes)          # boxes
            
        return image, targets


def create_data_loaders(data_yaml_path: str, batch_size: int = 16, img_size: int = 640, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """创建训练和验证数据加载器"""
    # 读取数据集配置
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 获取数据路径 - 修复路径拼接问题
    yaml_dir = Path(data_yaml_path).parent
    dataset_root = yaml_dir / Path(data_config.get('path', '.'))
    dataset_root = dataset_root.resolve() # 将路径转换为绝对路径，避免混淆

    train_images = dataset_root / data_config['train']
    val_images = dataset_root / data_config['val']
    
    # 推断标签路径 (YOLO标准：'images' 目录替换为 'labels')
    train_labels = Path(str(train_images).replace('images', 'labels'))
    val_labels = Path(str(val_images).replace('images', 'labels'))
    
    # 检查路径是否存在
    for path, name in [(train_images, '训练图片'), (val_images, '验证图片'),  
                       (train_labels, '训练标签'), (val_labels, '验证标签')]:
        if not path.exists():
            print(f"警告: {name}路径不存在: {path}")
    
    # 获取类别信息
    class_names = data_config.get('names', ['bacteria'])
    nc = data_config.get('nc', len(class_names))
    
    print("--- 数据集信息 ---")
    print(f"  数据集根目录: {dataset_root}")
    print(f"  类别数: {nc}")
    print(f"  类别名: {class_names}")
    print(f"  训练图片路径: {train_images}")
    print(f"  训练标签路径: {train_labels}")
    print(f"  验证图片路径: {val_images}")
    print(f"  验证标签路径: {val_labels}")
    print("--------------------")
    
    # 创建数据集
    train_dataset = BacteriaDataset(
        img_dir=str(train_images),
        label_dir=str(train_labels),
        img_size=img_size,
        augment=True,
        class_names=class_names
    )
    
    val_dataset = BacteriaDataset(
        img_dir=str(val_images),
        label_dir=str(val_labels),
        img_size=img_size,
        augment=False,
        class_names=class_names
    )
    
    # 检查数据集大小
    if len(train_dataset) == 0:
        raise ValueError(f"训练集为空！请检查路径: {train_images}")
    if len(val_dataset) == 0:
        print(f"警告: 验证集为空！路径: {val_images}")
    
    # <---  DataLoader 参数统一管理，并使用正确的 collate_fn --->
    loader_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': num_workers > 0,
        'collate_fn': collate_fn  
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
    
    dataset_info = {
        'nc': nc,
        'names': class_names,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }
    
    return train_loader, val_loader, dataset_info


if __name__ == "__main__":
    
   
    data_yaml = "dataset/yolo_format/dataset.yaml" 
    
    try:
        train_loader, val_loader, dataset_info = create_data_loaders(
            data_yaml, batch_size=4, num_workers=0
        )
        
        print(f"\n数据集信息: {dataset_info}")
        print(f"训练集批次数: {len(train_loader)}")
        if val_loader:
            print(f"验证集批次数: {len(val_loader)}")
        
        print("\n--- 测试一个训练批次 ---")
        # 测试一个批次
        images, targets = next(iter(train_loader))
        print(f"图片张量形状: {images.shape}")    # [batch_size, 3, img_size, img_size]
        print(f"目标张量形状: {targets.shape}")    # [total_boxes_in_batch, 6]
        print(f"目标示例 (前5个): \n{targets[:5]}")
        print("------------------------")
        
        # 验证批次索引是否正确
        assert targets[:, 0].min() == 0, "批次索引应该从0开始"
        assert targets[:, 0].max() == (4 - 1), "批次索引应该小于batch_size"
        print("批次索引验证通过！")

    except Exception as e:
        print(f"\n数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()