import os
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# 训练所需的图片尺寸要裁剪
def resize_and_save(input_dir, output_dir, size=(416, 416), show_samples=5):
    """
    将图像从中心裁剪/缩放到固定尺寸，并保存到新目录
    :param input_dir: 原始图片路径
    :param output_dir: 保存路径
    :param size: 目标尺寸 (w, h)
    :param show_samples: 随机显示的数量
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ 读取失败: {img_name}")
            continue

        h, w = img.shape[:2]

        # 居中裁剪为正方形
        min_side = min(h, w)
        top = (h - min_side) // 2
        left = (w - min_side) // 2
        crop_img = img[top: top + min_side, left: left + min_side]

        # 缩放到目标大小
        resized = cv2.resize(crop_img, size, interpolation=cv2.INTER_AREA)

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, resized)

    print(f"✅ 处理完成，共保存 {len(images)} 张到 {output_dir}")

    # 随机显示部分结果
    if len(images) > 0 and show_samples > 0:
        sample_imgs = random.sample(images, min(show_samples, len(images)))
        plt.figure(figsize=(12, 6))
        for i, img_name in enumerate(sample_imgs):
            img = Image.open(os.path.join(output_dir, img_name))
            plt.subplot(1, len(sample_imgs), i+1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(img_name)
        plt.show()


def main():
    input_dir = "F:\zywXM\PyCharmPro\yolov8-seg\data\JPEGImages"   # 修改为原始数据集路径
    output_dir = "F:\zywXM\PyCharmPro\yolov8-seg\data\JPEGImages"   # 修改为保存路径

    resize_and_save(input_dir, output_dir, size=(416, 416), show_samples=5)


if __name__ == "__main__":
    main()
