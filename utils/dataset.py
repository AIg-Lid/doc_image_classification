# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/13 14:16
@Auth ： Lid
@File ：dataset.py
@IDE ：PyCharm
"""
import glob
from PIL import Image

# 表格数据类别获取
table_class_list = []
with open(r"C:\Users\user3376\Desktop\common_image_classification\model\class.txt") as f:
    data = f.readlines()
    print("type_info: ", data)
    table_class_list.extend(data)


def exchang_png_to_jpg(images_path):
    # 先遍历没张图像的路径
    for img_path in images_path:
        img = Image.open(img_path)
        path_lst = img_path.split("\\")
        new_img_path = "/".join(path_lst[:-1]) + "/" + path_lst[-1][:-4] + ".jpg"
        img.save(new_img_path)


# 把中文图片名字路径中去掉
def reaname_png(images_path):
    # 先遍历没张图像的路径
    for ind, img_path in enumerate(images_path):
        img = Image.open(img_path).convert("RGB")
        path_lst = img_path.split("\\")
        new_img_path = "/".join(path_lst[:-1]) + "/" + "no_dot_" + str(ind) + ".jpg"
        img.save(new_img_path)


if __name__ == '__main__':

    image_paths = glob.glob(r"C:\Users\user3376\Desktop\vgg16_table_cls\data\stamp_image\*.png")
    exchang_png_to_jpg(image_paths[:])