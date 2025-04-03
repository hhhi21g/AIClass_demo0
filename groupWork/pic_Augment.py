import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import pytorch_lightning as pl
import seaborn as sns
import cv2 as cv
import numpy as np
import torch.nn.functional as F
from PIL import Image as Img

# 图像大小
IMAGE_SIZE = 900


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


# 定义一个数据增强变换，包括CLAHE转换、调整图像大小、随机仿射变换、转换为张量、归一化
tta_transform0 = transforms.Compose([
    CLAHE_Convert,  # CLAHE转换
    transforms.Resize(IMAGE_SIZE),  # 调整图像大小
    transforms.RandomAffine(degrees=(0, 45), translate=(0.05, 0.1), scale=(0.95, 1)),  # 随机仿射变换
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 归一化

# 定义一个数据增强的变换，包括CLAHE转换、调整图像大小、随机垂直翻转、随机水平翻转、颜色抖动、转换为张量、归一化
tta_transform1 = transforms.Compose([
    CLAHE_Convert,  # CLAHE转换
    transforms.Resize(IMAGE_SIZE),  # 调整图像大小
    transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转，翻转概率为0.5
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，翻转概率为0.5
    transforms.ColorJitter(brightness=0.2, contrast=0.01, saturation=0.2),  # 颜色抖动，亮度变化范围为0.2，对比度变化范围为0.01，饱和度变化范围为0.2
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])  # 归一化，均值和标准差分别为[0.485, 0.456, 0.406]和[0.229, 0.224, 0.225]

# 定义一个数据增强的变换，包括CLAHE转换、调整大小、随机应用随机裁剪、转换为张量、归一化
tta_transform2 = transforms.Compose([
    CLAHE_Convert,  # CLAHE转换
    transforms.Resize(IMAGE_SIZE),  # 调整大小
    transforms.RandomApply(transforms=  # 随机应用
                           [transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.4, 0.5),  # 随机裁剪
                                                         ratio=(1 / 3, 3), interpolation=  # 裁剪比例和插值方式
                                                         transforms.InterpolationMode.BICUBIC)], p=0.2),  # 随机应用的概率
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 归一化

# 定义一个列表，包含三个变换函数
tta_transforms = [tta_transform0,
                  tta_transform1,
                  tta_transform2]

# 读取示例图片
image_path = "example.jpg"  # 请替换为你的图片路径
original_img = Image.open(image_path).convert("RGB")

# 显示增强前后的对比图
fig, axes = plt.subplots(1, len(tta_transforms) + 1, figsize=(15, 5))

# 显示原始图片
axes[0].imshow(original_img)
axes[0].set_title("Original Image")
axes[0].axis("off")

# 依次应用不同的数据增强方法
for i, transform in enumerate(tta_transforms):
    augmented_img = transform(original_img)  # 变换后的图像
    augmented_img = augmented_img.permute(1, 2, 0).numpy()  # 转换为 NumPy 格式
    augmented_img = augmented_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # 反归一化
    augmented_img = np.clip(augmented_img, 0, 1)  # 限制数值范围

    axes[i + 1].imshow(augmented_img)
    axes[i + 1].set_title(f"Transform {i}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()
