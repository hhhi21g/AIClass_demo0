import cv2
import os
import numpy as np

# 获取picture文件夹中的所有文件
picture_folder = 'dataPre'
image_files = [f for f in os.listdir(picture_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# 创建equalized文件夹（如果不存在）
equalized_folder = 'dataPre\equalized'
if not os.path.exists(equalized_folder):
    os.makedirs(equalized_folder)

# 创建CLAHE对象
clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))

# 遍历每个图片文件
for image_file in image_files:
    # 读取图像
    image_path = os.path.join(picture_folder, image_file)
    image = cv2.imread(image_path)
    
    # 检查图像是否为彩色图像
    if len(image.shape) == 3:
        # 分离通道
        b, g, r = cv2.split(image)
        
        # 对每个通道应用CLAHE
        b_eq = clahe.apply(b)
        g_eq = clahe.apply(g)
        r_eq = clahe.apply(r)
        
        # 合并通道
        equalized_image = cv2.merge((b_eq, g_eq, r_eq))
    else:
        # 如果图像已经是灰度图像，则直接应用CLAHE
        equalized_image = clahe.apply(image)
    
    # 保存均衡化后的图像
    equalized_image_path = os.path.join(equalized_folder, image_file)
    cv2.imwrite(equalized_image_path, equalized_image)

print("图像处理完成并保存到文件夹：", equalized_folder)
