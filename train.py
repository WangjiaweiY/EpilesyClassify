import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from MRIDataset import MRIDataset
from Net import resnet3d18

def evaluate(model, dataloader, criterion, device):
    """
    在验证或测试阶段，计算平均损失
    """
    model.eval()  # 切换到评估模式
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)    # 输入形状: (B, 1, D, H, W)
            labels = labels.to(device)    # 标签形状: (B,)
            outputs = model(inputs)       # 输出形状: (B, num_classes)
            loss = criterion(outputs, labels)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据集及标签 CSV 路径（请根据实际情况调整）
    data_root = "../data"
    label_csv = '../data/副本2015-2023.08+癫痫全病例(2025.01.09)整理-发送.csv'
    dataset = MRIDataset(root_dir=data_root, label_csv=label_csv)

    # 将数据集划分为训练、验证、测试集（70%/15%/15%）
    total_num = len(dataset)
    train_num = int(total_num * 0.7)
    val_num = int(total_num * 0.15)
    test_num = total_num - train_num - val_num
    train_set, val_set, test_set = random_split(dataset, [train_num, val_num, test_num])
    
    # 创建 DataLoader（训练集打乱顺序，验证和测试集不打乱）
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False, num_workers=4)
    
    # 初始化模型、损失函数与优化器，并移到GPU
    model = resnet3d18(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()  # 训练模式
        running_loss = 0.0
        total_samples = 0
        
        # 训练阶段
        for inputs, labels in train_loader:
            inputs = inputs.to(device)   # (B, 1, D, H, W)
            labels = labels.to(device)   # (B,)
            
            optimizer.zero_grad()
            outputs = model(inputs)      # (B, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
        
        train_loss = running_loss / total_samples if total_samples > 0 else 0.0
        
        # 验证阶段
        val_loss = evaluate(model, val_loader, criterion, device)
        # 测试阶段
        test_loss = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
