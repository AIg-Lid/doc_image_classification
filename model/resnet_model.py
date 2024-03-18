# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/14 16:08
@Auth ： Lid
@File ：resnet_model.py
@IDE ：PyCharm
"""

import torch
from torchvision.models import *


class ResNet_ModelTrainer:
    def __init__(self, model_name, table_class_list):
        self.model_name = model_name
        self.table_class_list = table_class_list

    def create_model(self):
        """
        resnet系列模型选择
        """
        if self.model_name == 'resnet18':
            model = resnet18(pretrained=True)
        elif self.model_name == "resnet50":
            model = resnet50(pretrained=True)
        elif self.model_name == "resnet34":
            model = resnet34(pretrained=True)
        elif self.model_name == "resnet101":
            model = resnet101(pretrained=True)
        elif self.model_name == "resnet152":
            model = resnet152(pretrained=True)
        elif self.model_name == "resnext50_32x4d":
            model = resnext50_32x4d(pretrained=True)
        elif self.model_name == "resnext101_32x8d":
            model = resnext101_32x8d(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, len(self.table_class_list))
        return model

    def train_model(self):
        if self.model is None:
            print("Model has not been created. Please call create_model() first.")
            return

        print(f"Selected model: {self.model_name}")
        # 在这里添加模型训练的逻辑


if __name__ == '__main__':
    pass