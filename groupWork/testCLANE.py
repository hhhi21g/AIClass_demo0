import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image as Img
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
import timm
from tqdm import tqdm
import pytorch_lightning as pl
import seaborn as sns
import cv2 as cv
import numpy as np
import torch.nn.functional as F


def CLAHE_Convert(origin_input):
    # 创建CLAHE对象，clipLimit为对比度限制，tileGridSize为每个分块的大小
    clahe = cv.createCLAHE(clipLimit=40, tileGridSize=(10, 10))
    # 将输入图像转换为numpy数组
    t = np.asarray(origin_input)
    # 将图像从BGR颜色空间转换为HSV颜色空间
    t = cv.cvtColor(t, cv.COLOR_BGR2HSV)
    # 对HSV颜色空间中的V通道进行CLAHE处理
    t[:, :, -1] = clahe.apply(t[:, :, -1])
    # 将图像从HSV颜色空间转换回BGR颜色空间
    t = cv.cvtColor(t, cv.COLOR_HSV2BGR)
    # 将处理后的图像转换为PIL图像
    t = Img.fromarray(t)
    # 返回处理后的图像
    return t

image_path = 'shortData/test1.png'  # 替换为你自己的图像路径
original_image = Img.open(image_path)

# 使用CLAHE_Convert函数处理图像
processed_image = CLAHE_Convert(original_image)

# 保存处理后的图像
processed_image.save('processed1.jpg')  # 设置保存路径
