# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/14 21:55
@Auth ： Lid
@File ：predict.py
@IDE ：PyCharm
"""
import glob
import time
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from utils.dataset import table_class_list
from utils.parser import parser


class ModelPredict:
    # 初始化模型参数
    def __init__(self, args, table_class_list):
        self.table_class_list = table_class_list
        self.model_path = args.predict_model
        self.model = self.load_model()
        self.predicted_class = None
        self.predicted_prob = None

    def load_model(self):
        # 加载模型
        model = torch.load(self.model_path)
        model.eval()
        return model

    def preprocess_image(self, image):
        """
        图像预处理方法
        """

        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = data_transform(image).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        input_tensor = input_tensor.to(device)
        return input_tensor

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        # 加载要预测的图像
        input_tensor = self.preprocess_image(image)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        probilities = torch.nn.functional.softmax(outputs, dim=1)
        score = torch.max(probilities).item()
        _, pred = torch.max(outputs, 1)
        self.predicted_class = self.table_class_list[pred.item()]
        self.predicted_prob = score
        self.draw_image(image)
        return self.predicted_class

    def draw_image(self, image):
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 15)  # 选择适合的字体和大小
        cls = self.predicted_class.replace('\n', '')
        text = f"{cls} : {self.predicted_prob:.1f}"
        text_width, text_height = draw.textsize(text, font=font)
        text_position = (image.width - text_width - 10, 10)  # 右上角位置
        draw.text(text_position, text, fill=(0, 0, 255), font=font)
        image.save("output/d.jpg")


if __name__ == '__main__':

    args = parser.parse_args()
    # 进行预测
    model = ModelPredict(args, table_class_list)
    images_path = glob.glob(r'C:\Users\user3376\Desktop\common_image_classification\input\test\*.jpg')
    for img_path in images_path[1:4]:
        start_time = time.time()
        res = model.predict(img_path)
        print("eac image cost time: ", time.time() - start_time)
        print("image path: ", img_path)
        print("result: ", res, end="\n\n")
