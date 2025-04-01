import cv2
import os

def resize_image_opencv(input_path, output_path, target_size=(1024, 1024)):
    img = cv2.imread("test0.jpg")  # 读取图像
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)  # 调整大小
    cv2.imwrite(output_path, img_resized)  # 保存
    print(f"已调整尺寸并保存: {output_path}")

# 示例：
resize_image_opencv("test0.jpg", "output.jpg")
