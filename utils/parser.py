# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/14 13:53
@Auth ： Lid
@File ：parser.py
@IDE ：PyCharm
"""

import argparse
parser = argparse.ArgumentParser(description='image classification algorithm !')
parser.add_argument('--model', choices=['vgg16', 'resnet18', "resnet50", "mobilenetv3_large", "mobilenetv3_small..."], default='mobilenetv3_small', help='choose the model (default: vgg16)')
parser.add_argument('--predict_model',  default='checkpoint/mobilenetv3_small_model.pth', help='choose the model to predict')
parser.add_argument('--dataset',  default=r'data', help='dataset path')
parser.add_argument('--epochs',  default=10, help='num_epochs')
parser.add_argument('--lr',  default=0.001, help='learning rate')
parser.add_argument('--batch_size',  default=4)
parser.add_argument('--log_path',  default='log/training.log', help='log model train statement')
parser.add_argument('--log_interval',  default=4, help='print train info')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args.dataset)