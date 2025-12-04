from ultralytics import YOLO

# 验证阶段代码
def main():
    # 初始化模型
    model = YOLO("runs\\train\\potato\\weights\\best.pt") # 验证模型

# 验证模型
    results = model.val(
        data="potato.yaml",  # 数据集配置文件
        imgsz=416,  # 输入图像大小
        batch=4,  # 批大小，根据显存调整
        device=0  # 使用 GPU:0
    )
    print(results)

if __name__ == "__main__":
    main()
