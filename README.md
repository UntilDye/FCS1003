# 细菌检测项目 (Bacteria Detection)
基于YOLOv11的细菌检测系统，支持从自定义JSON格式数据到YOLO格式的转换、模型训练和推理。


## 项目结构
```
-bacteria-detection/
-│
-├── (core)/                      # 核心代码模块
-│   │
-│   ├── dataloader/            # 数据加载处理
-│   │   ├── data2yolo_format.py    → 数据格式转换(YOLO格式)
-│   │   └── data_loader.py         → 数据加载与预处理
-│   │
-│   ├── model/                 # 模型实现
-│   │   └── yolov11.py            → YOLOv11模型架构
-│   │
-│   └── utils/                 # 工具函数
-│       ├── loss.py               → 损失函数实现
-│       └── logger.py             → 训练日志记录
-│
-├── data/                      # 数据管理
-│   ├── raw/                   # 原始数据
-│   │   ├── Images/               → 原始图片集
-│   │   └── DatasetJson/          → JSON标注文件
-│   │
-│   └── processed/             # 处理后数据
-│       └── yolo_format/          → YOLO格式数据集
-│
-├── experiments/               # 实验记录
-│   └── runs/detect/              → 训练结果与检测输出
-│
-├── configs/                   # 配置管理
-│   └── train_config.yaml         → 训练超参数配置
-│
-├── scripts/                   # 可执行脚本
-│   ├── train.py                  → 模型训练入口
-│   └── inference.py              → 模型推理入口
-│
-└── docs/                      # 项目文档
-    ├── README.md                 → 项目说明文档
-    └── CHANGELOG.md              → 版本更新记录(忽略)

```

## 功能特性
```
-✅ 支持JSON到YOLO格式的数据转换
-✅ 完整的YOLOv11模型实现
-✅ 数据增强和自动混合精度训练
-✅ 灵活的损失函数（YOLOv11Loss / SimplifiedLoss）
-✅ 详细的训练日志和进度监控
-✅ WandB集成支持
-✅ 多种模型尺寸（n/s/m/l/x）
-✅ 早停和学习率调度
```
---
## 环境要求

bash


Python >= 3.8
torch >= 1.9.0
torchvision >= 0.10.0
opencv-python
albumentations
PIL
numpy
PyYAML
tqdm
pathlib
wandb (可选)
安装依赖
bash


pip install torch torchvision opencv-python albumentations pillow numpy pyyaml tqdm wandb

---
# 快速开始

# 1. 数据准备
原始数据格式
确保你的数据按以下结构组织：



dataset/
├── Images/           # 图片文件 (.jpg, .png, .jpeg)
└── DatasetJson/      # JSON标注文件
JSON标注格式
json


{
    "labels": [
        {
            "class": "colony",
            "x": 100,
            "y": 150,
            "width": 50,
            "height": 60
        }
    ]
}

---
# 2. 数据格式转换
将JSON格式转换为YOLO格式：





from dataloader.data2yolo_format import DatasetConverter

### 配置路径
IMAGE_DIR = "dataset/Images"
JSON_DIR = "dataset/DatasetJson"
OUTPUT_DIR = "dataset/yolo_format"

### 创建转换器
converter = DatasetConverter(IMAGE_DIR, JSON_DIR, OUTPUT_DIR, train_ratio=0.8)
converter.process_dataset()
或直接运行：




cd dataloader
python data2yolo_format.py

---
# 3. 配置训练参数
编辑 train_config.yaml：




### 模型配置
model_size: 'n'  # n, s, m, l, x
num_classes: 1   # 细菌检测类别数
pretrained_weights: null

### 训练配置
epochs: 50
batch_size: 16
img_size: 640
learning_rate: 0.0001
weight_decay: 0.0005
optimizer: 'AdamW'
scheduler: 'cosine'

---
### 数据配置
data_yaml: 'dataset/yolo_format/dataset.yaml'

### 输出配置
output_dir: 'runs/detect'
experiment_name: 'bacteria_detection'

---

## 4. 开始训练
bash


### 使用配置文件训练
python train.py --config train_config.yaml

### 使用命令行参数
python train.py \
    --data dataset/yolo_format/dataset.yaml \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --model-size n \
    --project runs/detect \
    --name bacteria_detection_v1

    ---
## 5. 训练监控
训练过程中会显示：

实时损失值和学习率
训练进度条
验证指标（如果有验证集）
自动保存最佳模型
训练结果保存在 runs/detect/实验名称/ 目录下：

mipsasm


runs/detect/bacteria_detection/
├── best.pt              # 最佳模型（YOLO格式）
├── last.pt              # 最新模型（YOLO格式）
├── best_custom.pt       # 最佳模型（自定义格式）
├── last_custom.pt       # 最新模型（自定义格式）
├── config.yaml          # 训练配置
└── train.log            # 训练日志
高级功能
使用WandB监控
在配置文件中启用WandB：



use_wandb: true
wandb_project: 'bacteria-detection'
恢复训练
bash

---
python train.py --resume runs/detect/bacteria_detection/last_custom.pt
自定义数据增强
在配置文件中调整增强参数：




augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  fliplr: 0.5

---
# 多GPU训练



### 使用DataParallel
CUDA_VISIBLE_DEVICES=0,1 python train.py --batch-size 32

### 使用DistributedDataParallel (推荐)
torchrun --nproc_per_node=2 train.py --batch-size 32


---

##    模型架构
### YOLOv11网络结构
Backbone: C2f模块组成的特征提取网络
Neck: FPN (Feature Pyramid Network) 多尺度特征融合
Head: 解耦的检测头，分别处理分类和回归
模型尺寸对比
模型	参数量	计算量(GFLOPs)	推理速度
n	2.6M	6.5	最快
s	9.4M	21.5	快
m	20.1M	48.0	中等
l	25.3M	64.6	慢
x	43.9M	108.1	最慢

---
## 数据格式说明
### YOLO格式转换
边界框坐标从绝对坐标转换为相对坐标
格式：class_id center_x center_y width height
所有坐标值归一化到[0,1]范围
数据集配置文件
自动生成的 dataset.yaml：




path: /path/to/dataset
train: images/train
val: images/val
nc: 1
names: ['colony']

---
# 故障排除
## 常见问题
CUDA内存不足：

减小batch_size
减小img_size
使用gradient_accumulation
数据加载失败

检查图片和标注文件是否配对：
确认文件路径正确
验证JSON格式
训练损失不收敛

降低学习率：
增加数据增强
检查标注质量
显存优化




## 在train.py中添加
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
性能优化建议
数据加载优化

## 调整num_workers数量
使用pin_memory=True
启用persistent_workers
训练加速

## 使用混合精度训练
启用compile模式（PyTorch 2.0+）
优化数据预处理管道

---
# API参考
DatasetConverter类



converter = DatasetConverter(
    image_dir="path/to/images",
    json_dir="path/to/json", 
    output_dir="path/to/output",
    train_ratio=0.8
)
BacteriaDataset类



dataset = BacteriaDataset(
    img_dir="path/to/images",
    label_dir="path/to/labels",
    img_size=640,
    augment=True,
    class_names=['bacteria']
)


## YOLOv11模型



model = create_model(
    model_size='n',
    num_classes=1,
    pretrained_weights='path/to/weights.pt'
)

---
# 贡献指南
##    Fork 项目
-创建特性分支 (git checkout -b feature/AmazingFeature)
-提交更改 (git commit -m 'Add some AmazingFeature')
-推送到分支 (git push origin feature/AmazingFeature)
-打开Pull Request
## 许可证
本项目基于MIT许可证开源。详见 LICENSE 文件（没有）。

---
# 联系方式
如有问题或建议，请通过以下方式联系：

## 提交Issue
发送邮件到：dontlike299@gmail.com
---
##    更新日志
v1.0.0 (2025-06-27)
###    初始版本发布
支持YOLOv11模型训练
实现JSON到YOLO格式转换
添加完整的训练管道
### 注意: 本项目专门针对细菌检测任务优化，如需用于其他目标检测任务，可能需要调整部分参数和配置。
