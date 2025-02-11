import os
import json
import argparse
import sys

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from utils import read_split_data
from my_dataset import MyDataSet
from modelV3 import swin_tiny_patch4_window7_224 as create_model

import torchvision

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_test_load(model_name):
    print('---test load name:', model_name)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 将输入数据尺寸调整为统一的224x224
        transforms.ToTensor(),  # 转换为tensor类型 后续加载入GPU计算
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
    ])
    dataset_test = torchvision.datasets.ImageFolder(r'E:\LiuYi\dataset\Remote_CL\test_taskB_english_index',  # 数据集已经被老师预处理好 按照类别整理好 使用ImageFolder即可创建dataset对象
                                                    transform=test_transforms,  # 使用train_transforms对数据调整
                                                    target_transform=None,
                                                    )
    test_loader = DataLoader(dataset_test,  # 使用pytorch自带的dataloader创建可迭代的数据输入
                             batch_size=512,  # 批大小
                             drop_last=False,  # 最后一个batch不足就舍弃
                             shuffle=False,  # 打乱
                             pin_memory=True,  # 将tensors拷贝到CUDA中的固定内存
                             num_workers=10)
    total = 0  # 数据总量
    correct = 0  # 数据中预测正确的个数

    loss_list_tes = []  # 用于保存验证集每个batch计算得出的loss 后续用于求均值
    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数定为交叉熵

    # 创建模型
    model = create_model(num_classes=20).to(device)
    # 加载模型权重
    # model_weight_path = "./"+model_name  # 选择模型最后一个epoch的训练权重
    model_weight_path = model_name  # 选择模型最后一个epoch的训练权重
    model.load_state_dict(torch.load(model_weight_path, map_location=device))  # 加载权重 并使用GPU加速

    model.to(device)  # 模型加载到GPU
    model.eval()  # 切换为评估模式 关掉dropout等等
    correct_cls = [0] * 20
    total_cls = [0] * 20
    acc_cls = []
    with torch.no_grad():  # 验证集停止网络参数更新
        for image, label in test_loader:  # 读取验证集图片及对应标签
            image = image.to(device)  # 图片加载到GPU
            label = label.to(device)  # 标签加载到GPU
            outputs = model(image)  # 模型计算图片的输出
            outputs = F.softmax(outputs, dim=1)  # 处理模型输出 使得概率最大的预测结果输出最大 输出总和为1
            predicted = torch.max(outputs, dim=1)[1]  # 选取输出最大的那个类别为最终预测结果
            total += label.size()[0]  # 累加得到总数据数量
            correct += (predicted == label).sum().cpu().item()  # 当模型输出和标签一致 正确的计数变量++

            c = (predicted == label).squeeze()
            for i in range(label.size()[0]):
                _label = label[i].item()
                correct_cls[_label] += c[i].item()
                total_cls[_label] += 1

        acc = correct * 1. / total  # 计算验证集数据预测正确率
        for i in range(20):
            acc_cls.append(correct_cls[i]/total_cls[i])
        acc2 = sum(correct_cls)/sum(total_cls)
        MA = sum(acc_cls)/20
        print('---Test: accuracy={:.4f} correct={} total={}'.format(acc, correct, total))
        print(correct_cls)
        print(total_cls)
        print(acc_cls)
        print('acc2, MA', acc2, MA)
    model.train()  # 模型切换为训练状态
    return acc  # 返回正确率及损失


if __name__ == '__main__':
    model_test_load('./yourmodel.pth')

