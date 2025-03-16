import os
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设定路径
train_dir = "data/train_images"
train_csv = "data/train_cultivar_mapping.csv"

# 读取训练数据 CSV 文件
train_df = pd.read_csv(train_csv)
train_df = train_df[train_df['image'] != '.DS_Store']
train_df.to_csv('data/train_cultivar_mapping.csv',index = False)
train_df = pd.read_csv(train_csv)

# 查看数据基本信息
print("数据集基本信息")
print(train_df.info())  # 显示数据列信息
print("\n前几行数据")
print(train_df.head())  # 显示前几行数据


plt.figure(figsize=(12, 6))
sns.countplot(y=train_df['cultivar'], order=train_df['cultivar'].value_counts().index)
plt.title("类别分布")
plt.xlabel("数量")
plt.ylabel("类别")
plt.show()


# 统计图像尺寸信息
def get_image_size(image_path):
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        return None  # 直接跳过非图片文件
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"无法读取 {image_path}: {e}")
        return None


image_sizes = [size for size in [get_image_size(os.path.join(train_dir, img)) for img in train_df['image']] if size is not None]
widths, heights = zip(*image_sizes)

# 显示图像尺寸分布
plt.figure(figsize=(12, 5))
sns.histplot(widths, bins=20, kde=True, label='Width', color='blue')
sns.histplot(heights, bins=20, kde=True, label='Height', color='orange')
plt.legend()
plt.title("图像宽高分布")
plt.xlabel("像素")
plt.ylabel("数量")
plt.show()

# 计算每个类别的样本数量
distribution = train_df['cultivar'].value_counts()
print("类别样本数量分布：")
print(distribution.describe())


# 预览部分图像
def plot_sample_images(df, image_folder, num_samples=12):
    sample_df = df.sample(num_samples, random_state=42)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        img_path = os.path.join(image_folder, row['image'])
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"{row['cultivar']}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


# 显示样本图像
plot_sample_images(train_df, train_dir)
