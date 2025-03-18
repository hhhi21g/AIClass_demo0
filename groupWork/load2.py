import cv2
import os
import numpy as np


def enhance_highlights(img):
    """专门针对高光区域增强的优化方案"""
    # 转换到Lab颜色空间处理亮度通道
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # 检测高光区域（亮度>220的区域）
    _, highlight_mask = cv2.threshold(l_channel, 220, 255, cv2.THRESH_BINARY)

    # 对高光区域进行特殊处理
    clahe_highlight = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))  # 低对比度限制
    clahe_normal = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # 常规区域处理

    # 分别处理高光区域和普通区域
    normal_region = clahe_normal.apply(l_channel)
    highlight_region = clahe_highlight.apply(l_channel)

    # 混合处理结果：仅在高光区域使用特殊处理
    enhanced_l = np.where(highlight_mask > 0, highlight_region, normal_region)
    a_eq = clahe_normal.apply(a)
    b_eq = clahe_normal.apply(b)
    # 合并通道并转回BGR
    enhanced_lab = cv2.merge([enhanced_l, a_eq, b_eq])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return result


# 创建输出文件夹
output_folder = 'dataPre/enhanced'
os.makedirs(output_folder, exist_ok=True)

# 处理所有图片
for image_file in os.listdir('dataPre'):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    img_path = os.path.join('dataPre', image_file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # 应用高光优化处理
    enhanced_img = enhance_highlights(img)

    # 保存结果
    cv2.imwrite(os.path.join(output_folder, image_file), enhanced_img)

print("高光优化处理完成，结果保存在:", output_folder)