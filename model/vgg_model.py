# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/14 16:07
@Auth ： Lid
@File ：vgg_model.py
@IDE ：PyCharm
"""

import torch
from torchvision.models import vgg19_bn, vgg19,vgg16_bn, vgg16, vgg13_bn, vgg13, vgg11_bn, vgg11


class VGG_ModelTrainer:
    def __init__(self, model_name, table_class_list):
        self.model_name = model_name
        self.table_class_list = table_class_list

    def create_model(self):
        if self.model_name == 'vgg11':
            model = vgg11(pretrained=True)
        elif self.model_name == 'vgg11_bn':
            model = vgg11_bn(pretrained=True)
        elif self.model_name == 'vgg13':
            model = vgg13(pretrained=True)
        elif self.model_name == 'vgg13_bn':
            model = vgg13_bn(pretrained=True)
        elif self.model_name == 'vgg16':
            model = vgg16(pretrained=True)
        elif self.model_name == 'vgg16_bn':
            model = vgg16_bn(pretrained=True)
        elif self.model_name == 'vgg19':
            model = vgg19(pretrained=True)
        elif self.model_name == "vgg19_bn":
            model = vgg19_bn(pretrained=True)
        else:
            pass

        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_features, len(self.table_class_list))
        return model

    def train_model(self):
        if self.model is None:
            print("Model has not been created. Please call create_model() first.")
            return

        print(f"Selected model: {self.model_name}")
        # 在这里添加模型训练的逻辑


if __name__ == '__main__':
    vgg16 = VGG_ModelTrainer("vgg16", ["1"])
    print("model: ", vgg16)