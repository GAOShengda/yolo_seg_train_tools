import cv2
import os
from ultralytics import YOLO

# 使用推理阶段代码
def predict_with_roi_folder():
    # 加载模型
    model = YOLO("runs/train/potato/weights/best.pt")

    # 图片输入路径和结果保存路径
    input_folder = r"F:\Desktop\JPEGImages"
    output_folder = os.path.join(input_folder, "reslut")
    os.makedirs(output_folder, exist_ok=True)  # 创建结果文件夹（不存在则创建）

    # 遍历文件夹中的图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # 支持的图片格式
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"图片加载失败: {filename}")
                continue

            h, w = img.shape[:2]

            # 中心 ROI 裁剪比例
            roi_scale = 0.6
            roi_w = int(w * roi_scale)
            roi_h = int(h * roi_scale)
            x1 = (w - roi_w) // 2
            y1 = (h - roi_h) // 2
            x2 = x1 + roi_w
            y2 = y1 + roi_h

            crop = img[y1:y2, x1:x2]

            # 推理
            results = model.predict(crop, imgsz=416, conf=0.25)

            # 保存结果
            annotated = results[0].plot()
            save_path = os.path.join(output_folder, f"res_{filename}")
            cv2.imwrite(save_path, annotated)
            print("已保存:", save_path)

if __name__ == "__main__":
    predict_with_roi_folder()
