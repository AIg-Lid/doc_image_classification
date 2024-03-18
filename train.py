# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/14 14:10
@Auth ： Lid
@File ：train.py
@IDE ：PyCharm
"""

import torch
import logging
import torchvision
from torchvision import transforms
from utils.parser import parser
from utils.dataset import table_class_list
from model.mobilenetv3_model import MobileNetV3_ModelTrainer
from model.vgg_model import VGG_ModelTrainer
from model.resnet_model import ResNet_ModelTrainer

args = parser.parse_args()

logging.basicConfig(filename=args.log_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("这是一个 {} 分类的图像算法任务！".format(len(table_class_list)))
logging.info(f"type_info: {table_class_list}")


# 训练模型
def train(model):

    """
    # 在上一步就可以传过来
    # model = resnet18(pretrained=True)

    # 冻结 VGG16 的参数，只训练分类器部分
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后的全连接层，以适应表格类型分类任务
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, len(table_class_list))
    """
    # 加载待分类的表格数据集，并进行预处理
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载表格数据集
    dataset = torchvision.datasets.ImageFolder(args.dataset,
                                               transform=data_transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()  # 设置为训练模式
        for batch_idx, (images, labels) in enumerate(train_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 统计训练数据的准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 打印训练进度
            if (batch_idx + 1) % args.log_interval == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, batch_idx + 1, len(train_data_loader),
                              running_loss / args.log_interval, 100.0 * correct / total))
                logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                             .format(epoch + 1, num_epochs, batch_idx + 1, len(train_data_loader),
                                     running_loss / args.log_interval, 100.0 * correct / total))
                running_loss = 0.0
                correct = 0
                total = 0

        # 在验证集上评估模型
        model.eval()  # 设置为评估模式
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_batch_idx, (val_images, val_labels) in enumerate(val_data_loader):
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()

                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()

        val_epoch_loss = val_loss / len(val_data_loader)
        val_accuracy = 100.0 * val_correct / val_total

        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
              .format(epoch + 1, num_epochs, val_epoch_loss, val_accuracy))
        logging.info('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
                     .format(epoch + 1, num_epochs, val_epoch_loss, val_accuracy))

    # 保存训练好的模型
    torch.save(model, 'checkpoint/' + args.model + "_model.pth")


# 主函数入口
def main():
    # 根据选择的模型创建相应的模型实例
    logging.info(f"Selected model: {args.model}")
    if "vgg" in args.model:
        trainer = VGG_ModelTrainer(args.model, table_class_list)
        model = trainer.create_model()
    elif "mobilenet" in args.model:
        trainer = MobileNetV3_ModelTrainer(args.model, table_class_list)
        model = trainer.create_model()
    else:
        trainer = ResNet_ModelTrainer(args.model, table_class_list)
        model = trainer.create_model()
    train(model)


if __name__ == '__main__':
    main()
