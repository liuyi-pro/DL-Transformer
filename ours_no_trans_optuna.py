

# 引入需要的官方库
import torch
import argparse
import os
import torchvision.transforms as transforms   # 使用transforms对输入的数据进行改变 多用于数据增强
import torchvision
from torch.utils.data import DataLoader # 使用DataLoader生成可迭代的batch数据
import modelV3
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import optuna
from modelV3 import swin_tiny_patch4_window7_224 as create_model  # 调用模型vit base
from torch.utils.tensorboard import SummaryWriter
import datetime


parser = argparse.ArgumentParser()
PC = 'taskA8030'  # 选择目前的机器

if PC == 'taskA8030':
    train_path = r"E:\LiuYi\dataset\Remote_CL\train_taskA"  # 对应机器的训练路径
    test_path = r"E:\LiuYi\dataset\Remote_CL\test_taskA"  # 对应机器的验证集路径
    parser.add_argument('--num_classes', type=int, default=15)  # 设定输出维度
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nw = 10  # num_worker
elif PC == 'taskB8030':
    train_path = r"E:\LiuYi\dataset\Remote_CL\train_taskB"  # 对应机器的训练路径
    test_path = r"E:\LiuYi\dataset\Remote_CL\test_taskB"  # 对应机器的验证集路径
    parser.add_argument('--num_classes', type=int, default=20)  # 设定输出维度
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nw = 10  # num_worker
parser.add_argument('--max_trails', type=int, default=20)  # 设定输出维度
parser.add_argument('--max_timing', type=int, default=60*60*24)  # 设定输出维度
parser.add_argument('--epochs', type=int, default=30)  # 训练迭代次数
parser.add_argument('--batch_size', type=int, default=150)  # 设定批大小
parser.add_argument('--lr_min', type=float, default=0.00001)  # 设定初始学习率
parser.add_argument('--lr_max', type=float, default=0.0001)  # 设定初始学习率
# parser.add_argument('--lrf', type=float, default=0.01)      # 没用
parser.add_argument('--model_name', default='', help='create model name')
parser.add_argument('--transfer_flag', type=bool, default=False)  # 不迁移自己的权重
# 预训练权重路径，如果不想载入就设置为空字符
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
# 是否冻结权重
parser.add_argument('--freeze_layers', type=bool, default=False)
# parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

args = parser.parse_args()


acc_max_global = 0  # 全局变量 当某个验证集正确率大于该值 保存当前模型 update这个值
optuna_trail_index = 0
model_accmax_for_test_name = ''
# 逃避otuna传参 并且为了避免obj每次切分不一致
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机切分
    transforms.RandomHorizontalFlip(),  # 左右翻转
    transforms.ToTensor(),          # 转换为tensor类型 后续加载入GPU计算
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])
dataset = torchvision.datasets.ImageFolder(train_path,  # 数据集已经被老师预处理好 按照类别整理好 使用ImageFolder即可创建dataset对象
                                                 transform=train_transforms,  # 使用train_transforms对数据调整
                                                 target_transform=None,
                                                 )
train_size = int(len(dataset) * 7 / 8)     # 裁切数据集 训练数据占总数据7/8
validate_size = len(dataset) - train_size  # 裁切数据集 验证数据占总数据1/8
train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size])
# 使用random_split将dataset随机裁切成训练集和验证集 两个数据集的数量给定 和为数据总量即可
train_loader = DataLoader(train_dataset,    # 使用pytorch自带的dataloader创建可迭代的数据输入
                          batch_size=args.batch_size,   # 批大小
                          drop_last=True,   # 最后一个batch不足就舍弃
                          shuffle=True,     # 打乱
                          pin_memory=True,  # 将tensors拷贝到CUDA中的固定内存
                          num_workers=nw)    # 6个进程来处理data loading
validate_loader = DataLoader(validate_dataset,  # 使用pytorch自带的dataloader创建可迭代的数据输入
                          batch_size=args.batch_size,       # 批大小
                          drop_last=True,       # 最后一个batch不足就舍弃
                          shuffle=True,         # 打乱
                          pin_memory=True,      # 将tensors拷贝到CUDA中的固定内存
                          num_workers=nw)        # 6个进程来处理data loading


def swin_test(model):
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 将输入数据尺寸调整为统一的224x224
        transforms.ToTensor(),  # 转换为tensor类型 后续加载入GPU计算
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
    ])
    dataset_test = torchvision.datasets.ImageFolder(test_path,  # 数据集已经被老师预处理好 按照类别整理好 使用ImageFolder即可创建dataset对象
                                               transform=test_transforms,  # 使用train_transforms对数据调整
                                               target_transform=None,
                                               )
    test_loader = DataLoader(dataset_test,  # 使用pytorch自带的dataloader创建可迭代的数据输入
                                 batch_size=args.batch_size,  # 批大小
                                 drop_last=True,  # 最后一个batch不足就舍弃
                                 shuffle=False,  # 打乱
                                 pin_memory=True,  # 将tensors拷贝到CUDA中的固定内存
                                 num_workers=nw)
    total = 0  # 数据总量
    correct = 0  # 数据中预测正确的个数
    loss_list_tes = []  # 用于保存验证集每个batch计算得出的loss 后续用于求均值
    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数定为交叉熵

    # 不再创建模型
    # model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # # 加载模型权重
    # model_weight_path = "./"+model_name  # 选择模型最后一个epoch的训练权重
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))  # 加载权重 并使用GPU加速

    model.to(device)  # 模型加载到GPU
    model.eval()  # 切换为评估模式 关掉dropout等等
    with torch.no_grad():  # 验证集停止网络参数更新
        for image, label in test_loader:  # 读取验证集图片及对应标签
            image = image.to(device)  # 图片加载到GPU
            label = label.to(device)  # 标签加载到GPU
            outputs = model(image)    # 模型计算图片的输出
            outputs = F.softmax(outputs, dim=1)  # 处理模型输出 使得概率最大的预测结果输出最大 输出总和为1
            predicted = torch.max(outputs, dim=1)[1]  # 选取输出最大的那个类别为最终预测结果
            total += label.size()[0]  # 累加得到总数据数量
            correct += (predicted == label).sum().cpu().item()  # 当模型输出和标签一致 正确的计数变量++

        acc = correct * 1. / total  # 计算验证集数据预测正确率

        print('---Test: accuracy={:.4f} correct={} total={}'.format(acc, correct, total))
    model.train()  # 模型切换为训练状态
    return acc  # 返回正确率及损失


def swin_test_load(model_name):
    print('---test load name:', model_accmax_for_test_name)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 将输入数据尺寸调整为统一的224x224
        transforms.ToTensor(),  # 转换为tensor类型 后续加载入GPU计算
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
    ])
    dataset_test = torchvision.datasets.ImageFolder(test_path,  # 数据集已经被老师预处理好 按照类别整理好 使用ImageFolder即可创建dataset对象
                                                    transform=test_transforms,  # 使用train_transforms对数据调整
                                                    target_transform=None,
                                                    )
    test_loader = DataLoader(dataset_test,  # 使用pytorch自带的dataloader创建可迭代的数据输入
                             batch_size=args.batch_size,  # 批大小
                             drop_last=True,  # 最后一个batch不足就舍弃
                             shuffle=False,  # 打乱
                             pin_memory=True,  # 将tensors拷贝到CUDA中的固定内存
                             num_workers=nw)
    total = 0  # 数据总量
    correct = 0  # 数据中预测正确的个数
    loss_list_tes = []  # 用于保存验证集每个batch计算得出的loss 后续用于求均值
    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数定为交叉熵

    # 创建模型
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # 加载模型权重
    model_weight_path = "./"+model_name  # 选择模型最后一个epoch的训练权重
    model.load_state_dict(torch.load(model_weight_path, map_location=device))  # 加载权重 并使用GPU加速

    model.to(device)  # 模型加载到GPU
    model.eval()  # 切换为评估模式 关掉dropout等等
    with torch.no_grad():  # 验证集停止网络参数更新
        for image, label in test_loader:  # 读取验证集图片及对应标签
            image = image.to(device)  # 图片加载到GPU
            label = label.to(device)  # 标签加载到GPU
            outputs = model(image)  # 模型计算图片的输出
            outputs = F.softmax(outputs, dim=1)  # 处理模型输出 使得概率最大的预测结果输出最大 输出总和为1
            predicted = torch.max(outputs, dim=1)[1]  # 选取输出最大的那个类别为最终预测结果
            total += label.size()[0]  # 累加得到总数据数量
            correct += (predicted == label).sum().cpu().item()  # 当模型输出和标签一致 正确的计数变量++

        acc = correct * 1. / total  # 计算验证集数据预测正确率

        print('---Test: accuracy={:.4f} correct={} total={}'.format(acc, correct, total))
    model.train()  # 模型切换为训练状态
    return acc  # 返回正确率及损失


def val(validate_loader, model):  # 定义验证集的测试方法
    total = 0  # 数据总量
    correct = 0  # 数据中预测正确的个数
    loss_list_val = []  # 用于保存验证集每个batch计算得出的loss 后续用于求均值
    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数定为交叉熵
    model.to(device)  # 模型加载到GPU
    model.eval()  # 切换为评估模式 关掉dropout等等
    with torch.no_grad():  # 验证集停止网络参数更新
        for image, label in validate_loader:  # 读取验证集图片及对应标签
            image = image.to(device)  # 图片加载到GPU
            label = label.to(device)  # 标签加载到GPU
            outputs = model(image)    # 模型计算图片的输出
            loss = criterion(outputs, label)  # 计算批损失
            outputs = F.softmax(outputs, dim=1)  # 处理模型输出 使得概率最大的预测结果输出最大 输出总和为1
            predicted = torch.max(outputs, dim=1)[1]  # 选取输出最大的那个类别为最终预测结果
            total += label.size()[0]  # 累加得到总数据数量
            correct += (predicted == label).sum().cpu().item()  # 当模型输出和标签一致 正确的计数变量++
            loss_list_val.append(loss.item())  # 保存计算得到的损失
        acc = correct * 1. / total  # 计算验证集数据预测正确率
        avg_loss = np.mean(loss_list_val)  # 损失求均值
        print('---VAL: loss={} accuracy={:.4f} correct={} total={}'.format(avg_loss, acc, correct, total))
    model.train()  # 模型切换为训练状态
    return acc, avg_loss  # 返回正确率及损失


def objective(trial):

    loss_train_mean_list = []  # 用于保存训练集每个epoch计算得出的平均loss 后续用于作图
    loss_val_mean_list = []    # 用于保存验证集每个epoch计算得出的平均loss 后续用于作图
    avg_loss_train = 0

    iter = 0     # 计数变量 实际没有用到
    acc_max_this_trail = 0  # 保存当前最大的验证集精度 新的结果超过改值会update这个值和保存当前的网络参数
    global acc_max_global
    global optuna_trail_index
    global model_accmax_for_test_name
    # model_accmax_for_test_name = ''  # 每个trail的acc max model名称需要记录 for test

    torch.manual_seed(666)       # cpu随机数种子，让结果可以复现
    torch.cuda.manual_seed(666)  # GPU随机数种子，让结果可以复现
    lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=False)
    print('LR', lr)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)  # 加载模型
    # model.load_state_dict(torch.load(args.weights))  # 头的数量不一致
    # model = torch.load(args.weights, map_location=device)  # bug ink21k没保存全部模型 只保存了权重

    if args.weights != "":
        if args.transfer_flag == False:  # no transfer
            print('transfer IMAGENET21k pre-train model')
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)  # 断言没触发
            weights_dict = torch.load(args.weights, map_location=device)["model"]  # IMAGENET21k pre-train model
            # 删除预训练参数中不需要的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    pg = [p for p in model.parameters() if p.requires_grad]

    model.to(device)             # 模型加载到GPU
    model.train()                # 打开模型训练开关 dropout等机制工作
    criterion = nn.CrossEntropyLoss().to(device)  # 指定loss函数为交叉熵
    # optimizer = optim.Adam(pg, lr=lr)  # 优化器定义为Adam二阶方法 vit
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)  # 定义学习率定期改变的计划
    optimizer = optim.AdamW(pg, lr=lr, weight_decay=5E-2)  # swinT
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)  # 定义学习率定期改变的计划 换成阶梯降低

    # 加入tensorboard
    curr_time = datetime.datetime.now()  # tensorboard
    timestamp = curr_time.date()  # tensorboard
    writer = SummaryWriter(str(PC) + '_'+"optuna_with_imagenet21K_runs" + "/" + 'LR' + str(lr) + '_BATCH' + str(args.batch_size) + '_EPOCH' + str(
        args.epochs) + '_CLASS' + str(args.num_classes) + '_FREEZE' + str(args.freeze_layers) + '_' + str(timestamp)+ '-' + str(curr_time.hour)+ '-' + str(curr_time.minute))  # tensorboard
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数 # tensorboard
    sample_num = 0
    # test_acc = 0
    for epoch in range(args.epochs):  # 循环训练epoch次
        loss_train_list = []    # 每次新的epoch训练开始前，清空上个epoch记录到的loss 方便记录此次epoch的平均loss
        for image, label in train_loader:  # 读取训练集图片及对应标签
            iter += 1                      # batch次数记录
            image = image.to(device)       # 图片加载到GPU
            label = label.to(device)       # 标签加载到GPU
            optimizer.zero_grad()          # 清空上个batch的梯度
            outputs = model(image)         # 模型计算图片的输出

            pred_classes = torch.max(outputs, dim=1)[1]  # tensorboard
            accu_num += torch.eq(pred_classes, label.to(device)).sum()  # tensorboard
            sample_num += image.shape[0]

            loss = criterion(outputs, label)  # 计算批损失
            loss.backward()                # 反向传播，计算当前梯度
            optimizer.step()               # 更新所有的参数
            loss_train_list.append(loss.item())  # 记录此次epoch的全部loss值
            # print('epoch:{} loss:{:.4f}'.format(epoch+1, loss.item()))

        train_acc = accu_num.item() / sample_num  # tensorboard

        scheduler.step()  # lr按照计划更新
        avg_loss_train = np.mean(loss_train_list)    # 计算此次epoch的平均loss
        loss_train_mean_list.append(avg_loss_train)  # 记录每个epoch的平均loss 后续可视化
        print('---Train: epoch={} loss={} accuracy={}'.format(epoch + 1, avg_loss_train, train_acc))
        acc_now, avg_loss_val = val(validate_loader, model)  # 每个epoch的模型训练结束就验证一次 观察loss是否训练完成 保存最佳模型
        loss_val_mean_list.append(avg_loss_val)  # 记录每个epoch的验证集平均loss 后续可视化

        if acc_now > acc_max_global:  # 当前epoch验证集正确率创了新高
            acc_now_str = str(acc_now).replace('.', '_')
            model_accmax_for_test_name = str(PC) + '_' \
                                        + 'IN21K_trans_optuna' + '_'  \
                                        + 'acc_max' + acc_now_str + '_' \
                                        + 'epoch' + str(epoch) + '_' \
                                        + 'trial' + str(optuna_trail_index) + '_' \
                                        + str(timestamp) + '-' + str(curr_time.hour) + '-' + str(curr_time.minute)+ '.pth'
            torch.save(model.state_dict(), model_accmax_for_test_name)  # 保存此时的模型参数
            # torch.save(model.state_dict(), 'val_model_acc_max_global.pth')  # 保存此时的模型参数
            acc_max_global = acc_now  # update最高正确率
            print('---acc_max_global update:', acc_max_global)
            test_acc = swin_test_load(model_accmax_for_test_name)
        else:
            test_acc = swin_test(model)
        writer.add_scalar("train_loss", avg_loss_train, epoch)  # tensorboard train—loss
        writer.add_scalar('train_acc', train_acc, epoch)  # tensorboard
        writer.add_scalar('val_loss', avg_loss_val, epoch)  # tensorboard
        writer.add_scalar('val_acc', acc_now, epoch)  # tensorboard
        writer.add_scalar('test_acc', test_acc, epoch)  # tensorboard
        writer.add_scalar('learning_rate', lr, epoch)  # tensorboard

        trial.report(acc_now, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # if acc_now > acc_max_this_trail:  # 当前epoch验证集正确率创了新高
        #     torch.save(model.state_dict(), 'val_model_acc_max_.pth')  # 保存此时的模型参数
        #     acc_max_this_trail = acc_now  # update最高正确率
        #     print('---acc_max update:', acc_max_this_trail)
        # else:
        #     print('---acc_max={} acc_now={}'.format(acc_max_this_trail, acc_now))
    optuna_trail_index += 1
    writer.close()  # tensorboard
    # torch.save(model.state_dict(), 'train_model_end.pth')  # 保存训练达到预定epoch时的网络模型 此模型一般不为最佳 仅做个对照实验
    return acc_max_global


if __name__ == '__main__':

    study = optuna.create_study(direction="maximize")  # 最大化正确率
    study.optimize(objective, n_trials=args.max_trails, timeout=args.max_timing)  # 最大尝试次数及最大尝试时间

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]  # 如果半数epoch过去 未超过之前尝试的中值 当前尝试中止 中止的次数
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]  # 完成尝试的次数统计

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))  # 一个开始了多少次尝试
    print("  Number of pruned trials: ", len(pruned_trials))  # 中止的次数
    print("  Number of complete trials: ", len(complete_trials))  # 完成的次数

    print("Best trial:")
    trial = study.best_trial  # 当前最好的尝试

    print("  Value: ", trial.value)  # 打印最高分类正确率

    print("  Params: ")
    for key, value in trial.params.items():  # 打印当前最好尝试的配置参数
        print("    {}: {}".format(key, value))

    swin_test_load(model_accmax_for_test_name)
