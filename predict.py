import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os
import re
from tqdm import tqdm

# RLE编解码函数
def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 预测专用数据集类 - 修改为返回高度和宽度作为单独值
class PredictionDataset(Dataset):
    def __init__(self, file_list, img_dir, transform=None):
        self.file_list = file_list
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 只获取文件名部分（第一列）
        full_line = self.file_list[idx]
        filename = full_line.split('\t')[0].strip()
        
        img_path = os.path.join(self.img_dir, filename)
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            # 尝试添加.jpg扩展名
            if not filename.lower().endswith(('.jpg', '.jpeg')):
                img_path_with_ext = os.path.join(self.img_dir, filename + '.jpg')
                image = cv2.imread(img_path_with_ext)
                if image is not None:
                    print(f"⚠️ 修正文件名: {filename} -> {filename}.jpg")
                    filename = filename + '.jpg'
                    img_path = img_path_with_ext
        
        if image is None:
            print(f"❌ 无法读取图像: {img_path}")
            # 创建替代图像
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            height, width = 512, 512
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # 返回高度和宽度作为单独值
        return image, filename, height, width

# 预测转换配置
def get_prediction_transform():
    return A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def main():
    # 配置参数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 4
    MODEL_PATH = './output/model/best_model.pth'
    TEST_A_DIR = './data/test_a'
    SUBMISSION_FILE = './data/test_a_samplesubmit.csv'
    OUTPUT_CSV = 'test_a_samplesubmit.csv'
    
    print("="*50)
    print(f"🚀 开始建筑物分割预测")
    print(f"📌 设备: {DEVICE}")
    print(f"📊 批次大小: {BATCH_SIZE}")
    print(f"🖼️ 测试图片文件夹: {TEST_A_DIR}")
    print(f"📄 提交样本文件: {SUBMISSION_FILE}")
    print("="*50)
    
    # 读取提交样本文件
    print("📂 读取提交样本文件...")
    with open(SUBMISSION_FILE, 'r') as f:
        submission_lines = [line.strip() for line in f.readlines()]
    
    # 创建数据集
    print("\n🖼️ 创建预测数据集...")
    transform = get_prediction_transform()
    prediction_dataset = PredictionDataset(
        file_list=submission_lines,
        img_dir=TEST_A_DIR,
        transform=transform
    )
    
    prediction_loader = DataLoader(
        prediction_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    # 加载模型
    print("\n🛠 加载模型...")
    model = smp.Unet(
        encoder_name='efficientnet-b4',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ 模型加载成功")
    
    # 进行预测（修改后的部分）
    print("\n🔮 开始预测...")
    results = []
    
    with torch.no_grad():
        for batch in tqdm(prediction_loader, desc="预测中"):
            images = batch[0].to(DEVICE)
            filenames = batch[1]
            heights = batch[2]  # 高度列表
            widths = batch[3]   # 宽度列表
            
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            for i in range(len(filenames)):
                h = heights[i].item() if torch.is_tensor(heights[i]) else heights[i]
                w = widths[i].item() if torch.is_tensor(widths[i]) else widths[i]
                
                pred_mask = preds[i].squeeze().cpu().numpy().astype(np.uint8)
                resized_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                rle = rle_encode(resized_mask) if np.sum(resized_mask) > 0 else ""
                results.append((filenames[i], rle))
    
    # 保存结果
    print("\n💾 保存预测结果...")
    with open(OUTPUT_CSV, 'w') as f:
        for i, (filename, rle) in enumerate(results):
            original_filename = submission_lines[i].split('\t')[0].strip()
            f.write(f"{original_filename}\t{rle}\n")
    
    print(f"✅ 预测结果已保存至: {OUTPUT_CSV}")
    
    # 打印结果统计
    print("\n预测结果示例:")
    for i in range(min(5, len(results))):
        print(f"{results[i][0]}\t{results[i][1]}")
    
    empty_count = sum(1 for _, rle in results if rle == "")
    print(f"\n无建筑物的图片数量: {empty_count}/{len(results)}")

if __name__ == '__main__':
    main()