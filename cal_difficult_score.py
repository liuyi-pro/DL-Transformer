import os
import re
import shutil
# predict
# （1-预测结果）+top5结果和ground truth的MSE or 交叉熵
# score越小越简单 score越大越复杂

import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from modelV3 import swin_tiny_patch4_window7_224 as create_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cal_correct_rate(img_path, model):

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    return predict.numpy(), predict_cla


def cal_score(predict, label):  # 正确率和MSE求和
    # MSE
    mse = np.mean((label - predict) ** 2)

    acc_hard = 1 - predict.max()

    score = acc_hard + mse
    # print("均方误差为：", mse)
    # print("acc_hard：", acc_hard)
    # print("score：", score)
    return score


def create_json(root: str):
    # 遍历文件夹，一个文件夹对应一个类别
    class_name = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    class_name.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(class_name))
    print(class_indices)
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    return len(class_name)


def main(path, model_dir):
    num_classes = create_json(path)  # 准备工作 生成json

    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = model_dir
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    for (root, dirs, files) in os.walk(path):  # 一共循环文件夹个数次 包含根目录
        dir_save = dirs
        for d in dirs:
            for file in os.listdir(os.path.join(root, d)):
                label = np.zeros(num_classes)
                # print(os.path.join(root, d) + '\\' + file)
                predict, predict_cla = cal_correct_rate(os.path.join(root, d) + '\\' + file, model)
                label[predict_cla] = 1
                # print(predict, predict_cla)
                score = cal_score(predict, label)
                # print(score)
                os.rename(os.path.join(root, d) + '\\' + file, os.path.join(root, d) + '\\' + str(score) + '_' + file)
                print(os.path.join(root, d) + '\\' + str(score) + '_' + file, 'OK')
            print('文件夹', d, '计算完了')


# 从path取文件计算score并另存为path2
def main2(path, path2, model_dir):
    num_classes = create_json(path)  # 准备工作 生成json

    # # 准备path2文件夹
    # for (root, dirs, files) in os.walk(path):  # 一共循环文件夹个数次 包含根目录
    #     for d in dirs:
    #         if os.path.exists(os.path.join(path2, d)):
    #             pass
    #         else:
    #             os.mkdir(os.path.join(path2, d))  # 新建课程各子类文件夹

    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = model_dir
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    for (root, dirs, files) in os.walk(path):  # 一共循环文件夹个数次 包含根目录
        dir_save = dirs
        for d in dirs:
            # 准备path2文件夹
            if os.path.exists(os.path.join(path2, d)):
                pass
            else:
                os.mkdir(os.path.join(path2, d))  # 新建课程各子类文件夹

            for file in os.listdir(os.path.join(root, d)):
                label = np.zeros(num_classes)
                # print(os.path.join(root, d) + '\\' + file)
                predict, predict_cla = cal_correct_rate(os.path.join(root, d) + '\\' + file, model)
                label[predict_cla] = 1
                # print(predict, predict_cla)
                score = cal_score(predict, label)
                # print(score)
                shutil.copy(os.path.join(root, d) + '\\' + file, os.path.join(path2, d) + '\\' + str(score) + '_' + file)
                # os.rename(os.path.join(root, d) + '\\' + file, os.path.join(path2, d) + '\\' + str(score) + '_' + file)
                print(os.path.join(path2, d) + '\\' + str(score) + '_' + file, 'OK')
            print('文件夹', d, '计算完了')


def cal_test():
    # create model
    model = create_model(num_classes=10).to(device)
    # load model weights
    model_weight_path = './weights/model.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    img = r'D:\Files_learning\5.writing things\desktop of TCSS\TCSS_Data22_0327\transfer\xx'
    predict, predict_cla = cal_correct_rate(img, model)
    label = np.zeros(10)
    label[predict_cla] = 1
    print(label)
    print(predict_cla)
    print(predict)
    score = cal_score(predict, label)
    print(score)


def name_recover(path):
    for (root, dirs, files) in os.walk(path):  # 一共循环文件夹个数次 包含根目录
        dir_save = dirs
        for d in dirs:
            index = 0
            for file in os.listdir(os.path.join(root, d)):

                os.rename(os.path.join(root, d) + '\\' + file, os.path.join(root, d) + '\\' + d + '_' + str(index) + '.jpg')
                index += 1
                print(os.path.join(root, d) + '\\' + d + '_' + str(index) + '.jpg', 'OK')


# 从path1复制number张图片到path2 从score小到大抽取
def CL_copy(path1, path2, number):
    for (root, dirs, files) in os.walk(path1):  # 一共循环文件夹个数次 包含根目录
        for d in dirs:
            if os.path.exists(os.path.join(path2, d)):
                pass
            else:
                os.mkdir(os.path.join(path2, d))  # 新建课程各子类文件夹

            num = number
            jpg_files = os.listdir(os.path.join(path1, d))  # 读入文件夹
            num_jpg = len(jpg_files)  # 统计文件夹中的文件个数

            for file in os.listdir(os.path.join(root, d)):
                shutil.copy(os.path.join(root, d) + '\\' + file, os.path.join(path2, d, file))
                num -= 1
                num_jpg -= 1
                if num <= 0 or num_jpg <= 0:
                    break
            print('文件夹', d, '复制完了')


# 因为python和windows文件名排序方式不一致 打印重命名后的文件夹内容 排查CL抽取的文件是不是正确
def check(path):
    for (root, dirs, files) in os.walk(path):  # 一共循环文件夹个数次 包含根目录
        for d in dirs:
            for file in os.listdir(os.path.join(root, d)):
                print(file)
            print('文件夹', d, '完了')


if __name__ == '__main__':
    PC = 'taskB_8030'  # 选择目前的机器

    if PC == 80572:
        train_path = r"D:\Files_learning\5.writing things\desktop of TCSS\TCSS_Data22_0327\transfer"  # 对应机器的训练路径
        test_path = r"D:\Files_learning\5.writing things\desktop of TCSS\TCSS_Data22_0327\transfer"  # 对应机器的验证集路径
        model_path = r'S:\Files_Learning\3_writing\31.defence\swin_transformer_remoteV3_1\80572_acc_max0_4125_epoch0_trial0_2023-08-21-16-20.pth'

    elif PC == 'taskB_8030':
        train_path = r"E:\LiuYi\dataset\Remote_CL\train_taskB"  # 训练集
        test_path = r"E:\LiuYi\dataset\Remote_CL\test_taskB"  # test集合不变
        model_path = r'E:\LiuYi\swin_transformer_remoteV3_1\yourfinetunemodel.pth'
        score_path = r'E:\LiuYi\dataset\Remote_CL\train_taskB_Score_Rename'
        CL0_path = r'E:\LiuYi\dataset\Remote_CL\train_taskB_CL0'
        CL1_path = r'E:\LiuYi\dataset\Remote_CL\train_taskB_CL1'
        CL2_path = r'E:\LiuYi\dataset\Remote_CL\train_taskB_CL2'



    # name_recover(train_path)  # 搭配main恢复名称 弃用
    # main(train_path, model_path)  # 弃用
    # main2(train_path, score_path, model_path)  # cal score 重命名并复制到新文件夹
    # cal_test()
    # print('\n CL0')
    # CL_copy(score_path, CL0_path, num1)  # CL0
    # print('\n CL1')
    # CL_copy(score_path, CL1_path, num1)
    print('\n CL2')
    CL_copy(score_path, CL2_path, yournum)
    # check(train_path)


