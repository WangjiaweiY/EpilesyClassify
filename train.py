import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from MRIDataset import MRIDataset
from Net import resnet3d18
from torch.utils.tensorboard import SummaryWriter
import models.resnet as resnet
import time
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

def generate_model(model_type='resnet', model_depth=50,
                   input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   pretrain_path = 'pretrain/resnet_50.pth',
                   num_classes=1):
    assert model_type in [
        'resnet'
    ]

    if model_type == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet.resnet10(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 256
    elif model_depth == 18:
        model = resnet.resnet18(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512
    elif model_depth == 34:
        model = resnet.resnet34(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512
    elif model_depth == 50:
        model = resnet.resnet50(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 101:
        model = resnet.resnet101(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 152:
        model = resnet.resnet152(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 200:
        model = resnet.resnet200(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048

    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                   nn.Linear(in_features=fc_input, out_features=num_classes, bias=True))

    if not no_cuda:
        if len(gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    print('loading pretrained model {}'.format(pretrain_path))
    pretrain = torch.load(pretrain_path)
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    # k 是每一层的名称，v是权重数值
    net_dict.update(pretrain_dict) #字典 dict2 的键/值对更新到 dict 里。
    model.load_state_dict(net_dict) #model.load_state_dict()函数把加载的权重复制到模型的权重中去

    print("-------- pre-train model load successfully --------")

    return model

def main():
    summaryWriter = SummaryWriter(log_dir='./logs/', flush_secs=30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(torch.cuda.get_device_name(0))

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
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    for x, y in train_loader:
        print(x.shape, y.shape)
        break
    
    # 初始化模型、损失函数与优化器，并移到GPU
    # model = resnet3d18(num_classes=2).to(device)
    model = generate_model(model_type='resnet', model_depth=50,
                   input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   pretrain_path = './pretrain/resnet_50_23dataset.pth',
                   num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    num_epochs = 10

    for epoch in range(num_epochs):
        start = time.time()
        per_epoch_loss = 0
        model.train()
        print("Epoch:", epoch + 1)

        with torch.enable_grad():
            total_correct = 0
            total_samples = 0
            per_epoch_loss = 0
            for x, y in tqdm(train_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

                batch_size = x.size(0)
                per_epoch_loss += loss.item() * batch_size
                total_samples += batch_size

                # 计算本batch的预测正确数量
                pred = y_hat.argmax(dim=1)  # 得到预测类别
                total_correct += (pred == y).sum().item()  # 统计预测正确的样本数

            train_loss = per_epoch_loss / total_samples if total_samples > 0 else 0.0
            train_acc = total_correct / total_samples if total_samples > 0 else 0.0
            print("Train Loss: {:.6f}\tTrain Acc: {:.6f}".format(train_loss, train_acc))
            summaryWriter.add_scalar('train_loss', train_loss, epoch)
            summaryWriter.add_scalar('train_accuracy', train_acc, epoch)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                batch_size = x.size(0)
                val_loss += loss.item() * batch_size
                total_samples += batch_size
                pred = y_hat.argmax(dim=1)
                val_correct += (pred == y).sum().item()
        val_loss = val_loss / total_samples if total_samples > 0 else 0.0
        val_acc = val_correct / total_samples if total_samples > 0 else 0.0

        print("Validation Loss: {:.6f}\tValidation Acc: {:.6f}".format(val_loss, val_acc))
        summaryWriter.add_scalar('val_loss', val_loss, epoch)
        summaryWriter.add_scalar('val_accuracy', val_acc, epoch)

        scheduler.step()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
