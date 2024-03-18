# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/14 16:08
@Auth ： Lid
@File ：mobilenetv3_model.py
@IDE ：PyCharm
"""
import torch
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


class MobileNetV3_ModelTrainer:
    def __init__(self, model_name, table_class_list):
        self.model_name = model_name
        self.table_class_list = table_class_list

    def create_model(self):
        if self.model_name == "mobilenetv3_large":
            model = mobilenet_v3_large(pretrained=True)
        elif self.model_name == 'mobilenetv3_small':
            model = mobilenet_v3_small(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_features, len(self.table_class_list))

        return model

    def train_model(self):
        if self.model is None:
            print("Model has not been created. Please call create_model() first.")
            return

        print(f"Selected model: {self.model_name}")
        # 在这里添加模型训练的逻辑

"""
def main():
    # 解析命令行参数...

    trainer = ModelTrainer(args.model, table_class_list)
    trainer.create_model()
    trainer.train_model()
"""
