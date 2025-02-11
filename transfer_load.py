import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from modelV3 import swin_tiny_patch4_window7_224 as create_model

# from my_dataset import MyDataSet
# from utils import read_split_data, train_one_epoch, evaluate  # 去掉的库

import datetime
import torchvision
import sys
from tqdm import tqdm  # 可视化训练进度


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    model.to(device)  # 模型加载到GPU

    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)  # 损失无穷报错
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.to(device)  # 模型加载到GPU
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = torchvision.datasets.ImageFolder(args.data_path_train,  # 文件夹方式读入数据
                                                     transform=data_transform["train"],  # 使用train_transforms对数据调整
                                                     target_transform=None,
                                                     )
    val_dataset = torchvision.datasets.ImageFolder(args.data_path_test,
                                                   transform=data_transform["val"],  # 使用train_transforms对数据调整
                                                   target_transform=None,  # 标注的预处理
                                                   )
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])
    #
    # # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 6])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    model = create_model(num_classes=args.num_classes).to(device)

    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     # weights_dict = torch.load(args.weights, map_location=device)["model"]  # imagenet22k
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     # 删除有关分类类别的权重
    #     for k in list(weights_dict.keys()):
    #         if "head" in k:
    #             del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))
    # 确认一下 他这么裁切预训练模型对不对  对的

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

        else:  # 如果是transfer 需要读取自己之前训练的模型
            print('transfer REMOTE pre-train model')
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)  # 断言没触发
            weights_dict = torch.load(args.weights, map_location=device)  # REMOTE pre-train model
            print(weights_dict.get('head.weight').shape)
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
        #  # 读取key尺寸是对的 说明print里面还是正确读取了模型参数
        # weight = model.state_dict()
        # print(weight.keys())
        # print(weight['head.weight'].shape)
        if model.state_dict().get('head.weight').shape[0] == args.num_classes:
            print(model.state_dict().get('head.weight').shape)
            print('pretrain model loads successfully, the number of classes is right')
        else:
            print(100 / 0)
        # 查看是不是成功加载模型预训练参数 直接看头的数量就行了 这里应该是num_classes=args.num_classes 而不是预训练模型的1000类

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)  # 定义学习率定期改变的计划 换成阶梯降低
    acc_max_val = 0
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        with torch.no_grad():
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        writer.add_scalar(tags[0], train_loss, epoch)
        writer.add_scalar(tags[1], train_acc, epoch)
        writer.add_scalar(tags[2], val_loss, epoch)
        writer.add_scalar(tags[3], val_acc, epoch)
        writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        scheduler.step()  # 优化器优化参数
        if val_acc > acc_max_val:  # 当前epoch验证集正确率创了新高
            torch.save(model.state_dict(), "./weights/model_val-{}.pth".format(epoch))  # 保存该模型的参数
            acc_max_val = val_acc  # update最高正确率
            print('---acc_max_val update:', acc_max_val)
        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    PC = 80571  # 选择目前的机器
    if PC == 8057:
        train_path = r"D:\Files_apps\BaiduNetdiskDownload\ML2dataset\project4\train"  # 对应机器的训练路径
        test_path = r"D:\Files_apps\BaiduNetdiskDownload\ML2dataset\project4\test"  # 对应机器的验证集路径
        parser.add_argument('--num_classes', type=int, default=35)  # 设定输出维度
    elif PC == 80571:
        train_path = r"D:\Files_learning\5.writing things\desktop of TCSS\TCSS_Data22_0327\200train"  # 对应机器的训练路径
        test_path = r"D:\Files_learning\5.writing things\desktop of TCSS\TCSS_Data22_0327\200test"  # 对应机器的验证集路径
        parser.add_argument('--num_classes', type=int, default=5)  # 设定输出维度
    elif PC == 8030:
        train_path = r"E:\LiuYi\dataset\RemoteSensing256_35classes\train"
        test_path = r"E:\LiuYi\dataset\RemoteSensing256_35classes\test"
        parser.add_argument('--num_classes', type=int, default=35)  # 设定输出维度
    elif PC == 80572:
        train_path = r"D:\Files_learning\5.writing things\desktop of TCSS\TCSS_Data22_0327\transfer"  # 对应机器的训练路径
        test_path = r"D:\Files_learning\5.writing things\desktop of TCSS\TCSS_Data22_0327\transfer"  # 对应机器的验证集路径
        parser.add_argument('--num_classes', type=int, default=10)  # 设定输出维度

    # parser.add_argument('--num_classes', type=int, default=5)   # 设定输出维度
    parser.add_argument('--epochs', type=int, default=3)  # 训练迭代次数
    parser.add_argument('--batch_size', type=int, default=16)  # 设定批大小
    parser.add_argument('--lr', type=float, default=0.0001)  # 设定初始学习率
    # parser.add_argument('--lrf', type=float, default=0.01)      # 没用
    parser.add_argument('--transfer_flag', type=bool, default=True)  # 设定初始学习率
    # 数据集所在根目录
    parser.add_argument('--data_path_train', type=str,
                        default=train_path)
    parser.add_argument('--data_path_test', type=str,
                        default=test_path)
    parser.add_argument('--model_name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str, default='./weights/model_val-0.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    curr_time = datetime.datetime.now()
    timestamp = curr_time.date()
    writer = SummaryWriter("runs" + "/" + 'LR' + str(opt.lr) + '_BATCH' + str(opt.batch_size) + '_EPOCH' + str(
        opt.epochs) + '_CLASS' + str(opt.num_classes) + '_FREEZE' + str(opt.freeze_layers) + '_' + str(timestamp))
    main(opt)

    writer.close()
