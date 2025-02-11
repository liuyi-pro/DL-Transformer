import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from modelV3 import swin_tiny_patch4_window7_224 as create_model
import datetime
import torchvision
import sys
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
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
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.to(device)
    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

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
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }

    # 实例化三个难度的训练数据集
    train_easy_dataset = torchvision.datasets.ImageFolder(args.data_path_train_easy,
                                                          transform=data_transform["train"])
    train_medium_dataset = torchvision.datasets.ImageFolder(args.data_path_train_medium,
                                                            transform=data_transform["train"])
    train_hard_dataset = torchvision.datasets.ImageFolder(args.data_path_train_hard,
                                                          transform=data_transform["train"])
    val_dataset = torchvision.datasets.ImageFolder(args.data_path_test,
                                                   transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 6])
    print('Using {} dataloader workers every process'.format(nw))

    train_easy_loader = torch.utils.data.DataLoader(train_easy_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw)
    train_medium_loader = torch.utils.data.DataLoader(train_medium_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      pin_memory=True,
                                                      num_workers=nw)
    train_hard_loader = torch.utils.data.DataLoader(train_hard_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(weights_dict.get('head.weight').shape)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
        print(model.state_dict().get('head.weight').shape)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    acc_max_val = 0
    current_phase = 0  # 0: easy, 1: medium, 2: hard
    phase_start_epoch = 0
    last_loss = float('inf')
    loss_patience = args.loss_patience
    loss_threshold = args.loss_threshold
    patience_counter = 0

    for epoch in range(args.epochs):
        # 选择当前阶段的数据加载器
        if current_phase == 0:
            current_loader = train_easy_loader
            print(f"Epoch {epoch}: Training with easy samples")
        elif current_phase == 1:
            current_loader = train_medium_loader
            print(f"Epoch {epoch}: Training with medium samples")
        else:
            current_loader = train_hard_loader
            print(f"Epoch {epoch}: Training with hard samples")

        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=current_loader,
                                                device=device,
                                                epoch=epoch)

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

        # 检查是否需要切换课程
        loss_decrease = (last_loss - train_loss) / last_loss if last_loss != 0 else 0
        if loss_decrease < loss_threshold:
            patience_counter += 1
        else:
            patience_counter = 0

        # 切换课程条件
        if (patience_counter >= loss_patience or epoch - phase_start_epoch >= 10) and current_phase < 2:
            current_phase += 1
            phase_start_epoch = epoch + 1
            patience_counter = 0
            print(f"Switching to {'medium' if current_phase == 1 else 'hard'} phase")

        last_loss = train_loss
        scheduler.step()

        if val_acc > acc_max_val:
            torch.save(model.state_dict(), "./weights/model_val-{}.pth".format(epoch))
            acc_max_val = val_acc
            print('---acc_max_val update:', acc_max_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 课程学习相关参数
    parser.add_argument('--loss_threshold', type=float, default=0.01,
                        help='threshold for loss decrease rate')
    parser.add_argument('--loss_patience', type=int, default=3,
                        help='number of epochs to wait before switching phase')

    # 三个难度数据集的路径
    parser.add_argument('--data_path_train_easy', type=str,
                        default=r"E:\LiuYi\dataset\Remote_CL\train_taskB_CL0")
    parser.add_argument('--data_path_train_medium', type=str,
                        default=r"E:\LiuYi\dataset\Remote_CL\train_taskB_CL1")
    parser.add_argument('--data_path_train_hard', type=str,
                        default=r"E:\LiuYi\dataset\Remote_CL\train_taskB_CL2")
    parser.add_argument('--data_path_test', type=str,
                        default=r"E:\LiuYi\dataset\Remote_CL\test_taskB")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    curr_time = datetime.datetime.now()
    timestamp = curr_time.date()
    writer = SummaryWriter("runs" + "/" + 'LR' + str(opt.lr) + '_BATCH' + str(opt.batch_size) + '_EPOCH' + str(
        opt.epochs) + '_CLASS' + str(opt.num_classes) + '_FREEZE' + str(opt.freeze_layers) + '_' + str(timestamp))

    main(opt)
    writer.close()