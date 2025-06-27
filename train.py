# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import wandb
from datetime import datetime
import os
import sys
import numpy as np
from tqdm import tqdm
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 修复导入路径
try:
    from datalodaer.data_loder import create_data_loaders
    from model.yolov11 import create_model, YOLOv11
    from utils.logger import setup_logger
    from utils.loss import SimplifiedLoss
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有必要的模块都存在")
    sys.exit(1)

class BacteriaDetectionTrainer:
    def __init__(self, config):
        """
        细菌检测训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir']) / config.get('experiment_name', 'bacteria_detection')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(self.output_dir / 'train.log')
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化损失函数
        self.criterion = SimplifiedLoss()
        
        # 初始化wandb（可选）
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'bacteria-detection'),
                name=config.get('experiment_name', f"bacteria-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                config=config
            )
    
    def _create_model(self):
        """创建模型"""
        model_size = self.config.get('model_size', 'n')
        num_classes = self.config.get('num_classes', 1)
        pretrained_weights = self.config.get('pretrained_weights', None)
        
        model = create_model(
            model_size=model_size,
            num_classes=num_classes,
            pretrained_weights=pretrained_weights
        )
        
        model = model.to(self.device)
        
        # 统计参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型大小: {model_size}")
        self.logger.info(f"类别数: {num_classes}")
        self.logger.info(f"模型参数总数: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        
        return model
    
    def _setup_optimizer_scheduler(self):
        """设置优化器和学习率调度器"""
        # 分组参数 - 对不同层使用不同的权重衰减
        pg0, pg1, pg2 = [], [], []  # 优化器参数组
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if '.bias' in name:
                    pg2.append(param)  # bias (no decay)
                elif 'bn' in name or 'norm' in name:
                    pg1.append(param)  # BN weights (no decay)
                else:
                    pg0.append(param)  # conv weights (with decay)
        
        optimizer_type = self.config.get('optimizer', 'AdamW')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0005)
        
        if optimizer_type == 'AdamW':
            optimizer = optim.AdamW([
                {'params': pg0, 'weight_decay': weight_decay},
                {'params': pg1, 'weight_decay': 0.0},
                {'params': pg2, 'weight_decay': 0.0}
            ], lr=lr, betas=(0.9, 0.999), eps=1e-8)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD([
                {'params': pg0, 'weight_decay': weight_decay},
                {'params': pg1, 'weight_decay': 0.0},
                {'params': pg2, 'weight_decay': 0.0}
            ], lr=lr, momentum=0.937, nesterov=True)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 学习率调度器
        scheduler_type = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 100)
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=lr * 0.01
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=epochs // 3,
                gamma=0.1
            )
        elif scheduler_type == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=epochs,
                pct_start=0.1
            )
        else:
            scheduler = None
        
        self.logger.info(f"优化器: {optimizer_type}")
        self.logger.info(f"学习率调度器: {scheduler_type}")
        self.logger.info(f"初始学习率: {lr}")
        
        return optimizer, scheduler
    
    def _setup_dataloader(self):
        """设置数据加载器"""
        data_yaml_path = self.config.get('data_yaml')
        if not data_yaml_path or not Path(data_yaml_path).exists():
            self.logger.error(f"数据集配置文件不存在: {data_yaml_path}")
            return None, None, None
        
        batch_size = self.config.get('batch_size', 16)
        img_size = self.config.get('img_size', 640)
        
        try:
            train_loader, val_loader, dataset_info = create_data_loaders(
                data_yaml_path=data_yaml_path,
                batch_size=batch_size,
                img_size=img_size,
                num_workers=min(4, os.cpu_count())  # 自动调整工作进程数
            )
            
            self.logger.info(f"数据集加载成功: {dataset_info}")
            return train_loader, val_loader, dataset_info
            
        except Exception as e:
            self.logger.error(f"数据加载器创建失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100)
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播
            try:
                outputs = self.model(images)
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"检测到无效损失值: {loss.item()}")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{total_loss/(batch_idx+1):.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
            except Exception as e:
                self.logger.error(f"训练步骤出错 (batch {batch_idx}): {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, val_loader):
        """验证模型"""
        if val_loader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation', ncols=100):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                try:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                except Exception as e:
                    self.logger.warning(f"验证步骤出错: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def save_checkpoint(self, epoch, optimizer, scheduler, best_loss, train_loss, val_loss, is_best=False):
        """保存检查点"""
        # 标准检查点格式
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_loss': best_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # YOLO 兼容格式
        yolo_checkpoint = {
            'model': self.model,  # 完整模型
            'epoch': epoch,
            'best_fitness': 1.0 / (best_loss + 1e-6),  # 转换为适应度
            'optimizer': optimizer.state_dict(),
            'train_args': self.config,
            'date': datetime.now().isoformat(),
            'version': '11.0.0'
        }
        
        # 保存标准格式（用于恢复训练）
        torch.save(checkpoint, self.output_dir / 'last_custom.pt')
        
        # 保存 YOLO 兼容格式（用于推理）
        torch.save(yolo_checkpoint, self.output_dir / 'last.pt')
        
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_custom.pt')
            torch.save(yolo_checkpoint, self.output_dir / 'best.pt')
            self.logger.info(f"保存最佳模型: epoch {epoch}, loss {best_loss:.4f}")
        
        # 定期保存
        save_period = self.config.get('save_period', 10)
        if epoch % save_period == 0:
            torch.save(checkpoint, self.output_dir / f'epoch_{epoch}_custom.pt')
            torch.save(yolo_checkpoint, self.output_dir / f'epoch_{epoch}.pt')
    
    def train(self):
        """开始训练"""
        self.logger.info("="*50)
        self.logger.info("开始训练细菌检测模型")
        self.logger.info("="*50)
        
        # 设置优化器和调度器
        optimizer, scheduler = self._setup_optimizer_scheduler()
        
        # 设置数据加载器
        train_loader, val_loader, dataset_info = self._setup_dataloader()
        
        if train_loader is None:
            self.logger.error("数据加载失败，训练终止")
            return None
        
        epochs = self.config.get('epochs', 100)
        best_loss = float('inf')
        patience = self.config.get('patience', 50)
        patience_counter = 0
        
        # 保存配置文件
        with open(self.output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        try:
            for epoch in range(1, epochs + 1):
                self.logger.info(f"\nEpoch {epoch}/{epochs}")
                self.logger.info("-" * 30)
                
                # 训练
                train_loss = self.train_epoch(train_loader, optimizer, epoch)
                
                # 验证
                val_loss = self.validate(val_loader) if val_loader else None
                
                # 更新学习率
                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                        scheduler.step()
                    else:
                        scheduler.step()
                
                # 当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                # 记录日志
                log_msg = f"Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}"
                
                self.logger.info(log_msg)
                
                # 记录到wandb
                if self.config.get('use_wandb', False):
                    log_dict = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'learning_rate': current_lr
                    }
                    if val_loss is not None:
                        log_dict['val_loss'] = val_loss
                    wandb.log(log_dict)
                
                # 确定用于早停的损失
                monitor_loss = val_loss if val_loss is not None else train_loss
                
                # 保存检查点
                is_best = monitor_loss < best_loss
                if is_best:
                    best_loss = monitor_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                self.save_checkpoint(epoch, optimizer, scheduler, best_loss, train_loss, val_loss, is_best)
                
                # 早停检查
                if patience_counter >= patience:
                    self.logger.info(f"验证损失在 {patience} 个epoch内没有改善，提前停止训练")
                    break
            
            self.logger.info("="*50)
            self.logger.info("训练完成!")
            self.logger.info(f"最佳损失: {best_loss:.4f}")
            self.logger.info(f"模型保存在: {self.output_dir}")
            self.logger.info("="*50)
            
            if self.config.get('use_wandb', False):
                wandb.log({"best_loss": best_loss})
                wandb.finish()
            
            return {
                'best_loss': best_loss,
                'final_epoch': epoch,
                'output_dir': str(self.output_dir)
            }
            
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
            self.save_checkpoint(epoch, optimizer, scheduler, best_loss, train_loss, val_loss or 0.0)
            if self.config.get('use_wandb', False):
                wandb.finish()
            return None
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            if self.config.get('use_wandb', False):
                wandb.finish()
            raise e

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='细菌检测模型训练')
    parser.add_argument('--config', type=str, default='train_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data', type=str, default='dataset/yolo_format/dataset.yaml',
                       help='数据集YAML文件路径')
    parser.add_argument('--epochs', type=int,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int,
                       help='批次大小')
    parser.add_argument('--img-size', type=int,
                       help='图片尺寸')
    parser.add_argument('--model-size', type=str, choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小')
    parser.add_argument('--num-classes', type=int,
                       help='类别数量')
    parser.add_argument('--project', type=str,
                       help='项目保存目录')
    parser.add_argument('--name', type=str,
                       help='实验名称')
    parser.add_argument('--pretrained', type=str,
                       help='预训练权重路径')
    parser.add_argument('--resume', type=str,
                       help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"加载配置文件: {args.config}")
    else:
        config = {}
        print("使用默认配置")
    
    # 命令行参数覆盖配置文件
    if args.data:
        config['data_yaml'] = args.data
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.img_size:
        config['img_size'] = args.img_size
    if args.model_size:
        config['model_size'] = args.model_size
    if args.num_classes:
        config['num_classes'] = args.num_classes
    if args.project:
        config['output_dir'] = args.project
    if args.name:
        config['experiment_name'] = args.name
    if args.pretrained:
        config['pretrained_weights'] = args.pretrained
    
    # 设置默认值
    config.setdefault('output_dir', 'runs/detect')
    config.setdefault('experiment_name', 'bacteria_detection')
    config.setdefault('num_classes', 1)
    config.setdefault('model_size', 'n')
    config.setdefault('epochs', 100)
    config.setdefault('batch_size', 16)
    config.setdefault('img_size', 640)
    config.setdefault('data_yaml', 'dataset/yolo_format/dataset.yaml')
    
    print("最终配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建训练器并开始训练
    trainer = BacteriaDetectionTrainer(config)
    results = trainer.train()
    
    if results:
        print(f"\n训练成功完成!")
        print(f"最佳损失: {results['best_loss']:.4f}")
        print(f"输出目录: {results['output_dir']}")
    else:
        print("训练未完成或失败")

if __name__ == "__main__":
    main()