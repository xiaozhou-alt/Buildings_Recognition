import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import os
import re

# RLE编解码函数 (已增强)
def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(512, 512)):
    if pd.isna(mask_rle) or not isinstance(mask_rle, str) or mask_rle.strip() == '':
        return np.zeros(shape, dtype=np.uint8)
    
    # 清理RLE编码中的额外制表符
    mask_rle = re.sub(r'\t+', ' ', mask_rle.strip())
    
    s = mask_rle.split()
    try:
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape, order='F')
    except Exception as e:
        print(f"Error decoding RLE: {e}\nRLE string: '{mask_rle}'")
        return np.zeros(shape, dtype=np.uint8)

# 自定义数据集类（已修复）
class BuildingDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, mode='train'):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        
        # 清理文件名 - 移除制表符和空格
        self.df['name'] = self.df['name'].apply(lambda x: re.sub(r'\t+', '', str(x).strip()))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx, 0]
        # 确保文件名有正确的扩展名
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            filename += '.jpg'
            
        img_path = os.path.join(self.img_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            # 尝试不同的大小写
            img_dir_files = os.listdir(self.img_dir)
            matched_files = [f for f in img_dir_files if f.lower() == filename.lower()]
            
            if matched_files:
                img_path = os.path.join(self.img_dir, matched_files[0])
                print(f"⚠️ 修正文件名大小写: {filename} -> {matched_files[0]}")
            else:
                print(f"⚠️ 文件不存在: {img_path}")
                # 创建替代图像
                image = np.zeros((512, 512, 3), dtype=np.uint8)
                if self.mode == 'train' or self.mode == 'val':
                    mask = np.zeros((512, 512), dtype=np.float32)
                    if self.transform:
                        augmented = self.transform(image=image, mask=mask)
                        return augmented['image'], augmented['mask']
                    return image, mask
                else:
                    if self.transform:
                        augmented = self.transform(image=image)
                        return augmented['image']
                    return image
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 无法读取图像: {img_path}")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == 'train' or self.mode == 'val':
            mask_rle = self.df.iloc[idx, 1]
            mask = rle_decode(mask_rle, shape=image.shape[:2])
            mask = mask.astype(np.float32)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            return image, mask
        
        else:  # 测试模式
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image

# 数据增强配置
def get_train_transform():
    return A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# Dice系数计算函数
def dice_coeff(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# 模型训练函数（添加早停机制）
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=15, patience=5):
    best_val_dice = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_dice': []}
    no_improve_count = 0  # 记录没有提升的epoch数
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # 训练阶段
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, 
                                                      desc=f"Epoch {epoch+1}/{num_epochs} - Training",
                                                      total=len(train_loader))):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # 每100个batch打印一次进度
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        # 验证阶段
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating", total=len(val_loader)):
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1)
                
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                val_dice += dice_coeff(preds.float(), masks).item()
        
        val_dice /= len(val_loader)
        history['val_dice'].append(val_dice)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # 检查是否有提升
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            no_improve_count = 0
            torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            print(f"💾 保存新的最佳模型，Dice: {best_val_dice:.4f}")
        else:
            no_improve_count += 1
            print(f"🔄 验证集Dice未提升，连续 {no_improve_count}/{patience} 个epoch")
            
            # 检查是否达到早停条件
            if no_improve_count >= patience:
                print(f"🛑 早停触发！连续 {patience} 个epoch验证集Dice未提升。")
                print(f"🏆 最佳验证集Dice: {best_val_dice:.4f} (epoch {best_epoch+1})")
                break
    
    # 如果正常完成所有epoch，打印最佳结果
    if no_improve_count < patience:
        print(f"✅ 训练完成！最佳验证集Dice: {best_val_dice:.4f} (epoch {best_epoch+1})")
    
    return history

# 主程序
def main():
    # 配置参数
    BATCH_SIZE = 8
    LR = 0.001
    EPOCHS = 50
    PATIENCE = 5  # 早停耐心值
    IMG_SIZE = 384
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*50)
    print(f"🚀 启动建筑物分割训练")
    print(f"📌 设备: {DEVICE}")
    print(f"📊 批次大小: {BATCH_SIZE}")
    print(f"🎯 学习率: {LR}")
    print(f"🔄 训练轮数: {EPOCHS}")
    print(f"⏱️ 早停耐心值: {PATIENCE}")
    print("="*50)
    
    # 加载数据 - 使用制表符分隔，处理空值
    print("📂 加载数据...")
    train_df = pd.read_csv(
        '/kaggle/input/buildings-recognition-processed/train.csv', 
        header=None, 
        sep='\t',  # 使用制表符分隔
        names=['name', 'mask'],
        on_bad_lines='warn'  # 忽略格式不正确的行
    )
    test_df = pd.read_csv(
        '/kaggle/input/buildings-recognition-processed/test.csv', 
        header=None, 
        sep='\t',  # 使用制表符分隔
        names=['name', 'mask'],
        on_bad_lines='warn'
    )
    
    # 处理空值 - 对于没有建筑物的图像，mask为空
    train_df['mask'] = train_df['mask'].fillna('')
    test_df['mask'] = test_df['mask'].fillna('')
    
    # 清理数据 - 移除文件名中的制表符
    train_df['name'] = train_df['name'].apply(lambda x: re.sub(r'\t+', '', str(x).strip()))
    test_df['name'] = test_df['name'].apply(lambda x: re.sub(r'\t+', '', str(x).strip()))
    
    # 打印样本
    print("\n训练数据样本:")
    print(train_df.head())
    print(f"\n训练集大小: {len(train_df)}")
    print(f"没有建筑物的图像数量: {train_df['mask'].eq('').sum()}")
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}")
    
    # 创建数据集和数据加载器
    train_dataset = BuildingDataset(
        train_df, 
        img_dir='/kaggle/input/buildings-recognition-processed/train', 
        transform=get_train_transform(), 
        mode='train'
    )
    val_dataset = BuildingDataset(
        val_df, 
        img_dir='/kaggle/input/buildings-recognition-processed/train', 
        transform=get_val_transform(), 
        mode='val'
    )
    test_dataset = BuildingDataset(
        test_df, 
        img_dir='/kaggle/input/buildings-recognition-processed/test', 
        transform=get_val_transform(), 
        mode='test'
    )
    
    # 验证文件路径
    sample_idx = 0
    sample_file = os.path.join('/kaggle/input/buildings-recognition-processed/train', train_df.iloc[sample_idx, 0])
    print(f"\n验证文件路径: {sample_file}")
    print(f"文件存在: {os.path.exists(sample_file)}")
    if os.path.exists(sample_file):
        print(f"文件大小: {os.path.getsize(sample_file)} bytes")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 初始化模型
    print("\n🛠 初始化模型...")
    model = smp.Unet(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)
    
    # 打印模型信息
    print(f"模型架构: U-Net with EfficientNet-B4")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params/1e6:.2f}M")
    
    # 损失函数和优化器
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 训练模型
    print("\n🔥 开始训练...")
    history = train_model(
        model, train_loader, val_loader,
        optimizer, criterion, DEVICE, 
        num_epochs=EPOCHS, patience=PATIENCE
    )
    
    # 测试集评估
    print("\n🧪 测试集评估...")
    model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
    model.eval()
    
    test_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="测试中", total=len(val_loader)):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).unsqueeze(1)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            test_dice += dice_coeff(preds.float(), masks).item()
    
    test_dice /= len(val_loader)
    print(f"✅ 测试集Dice系数: {test_dice:.4f}")
    
    # 可视化5个测试样本
    print("\n🎨 可视化预测结果...")
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(5, 3, figsize=(15, 25))
        
        # 确保测试集有足够样本
        if len(test_dataset) < 5:
            sample_indices = range(len(test_dataset))
        else:
            sample_indices = np.random.choice(len(test_dataset), 5, replace=False)
        
        for i, idx in enumerate(sample_indices):
            image = test_dataset[idx]
            
            if isinstance(image, tuple):  # 如果测试集包含mask
                image_tensor, true_mask = image
                true_mask = true_mask.cpu().numpy()
            else:
                image_tensor = image
                true_mask = None
            
            # 预测
            input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_tensor)
            pred_mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy()
            
            # 反标准化图像
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)
            
            # 显示结果
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f'origin: {test_df.iloc[idx, 0]}')
            axes[i, 0].axis('off')
            
            if true_mask is not None:
                axes[i, 1].imshow(true_mask, cmap='gray')
                axes[i, 1].set_title('true_mask')
            else:
                axes[i, 1].axis('off')
                axes[i, 1].set_title('no_mask')
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('predict_mask')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('/kaggle/working/predictions.png')
        plt.show()
        print("📸 预测结果已保存为 /kaggle/working/predictions.png")

if __name__ == '__main__':
    main()