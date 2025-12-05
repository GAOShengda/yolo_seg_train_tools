from ultralytics import YOLO

# 训练阶段代码
def main():
    # 初始化模型
    model = YOLO("yolov8s-seg.pt") # 开始训练

# 开始训练
    model.train(
    data="potato.yaml",  # 数据集配置文件
    epochs=100,  # 训练轮数
    imgsz=416,  # 输入图像尺寸
    batch=8,  # 批大小，根据显存调整
    device=0,  # GPU 0
    project="runs/train",  # 保存训练结果的目录
    name="potato",  # 本次训练名称
    exist_ok=True,  # 如果目录已存在允许覆盖
    cache=True,  # 缓存数据加快训练
    save_period=1,  # 每轮保存权重
    box=10,  # 提高框回归权重
    mask_ratio=2,  # 提高mask精度
    patience=200,  # 早停轮数
    # verbose=True,  # 实时显示训练信息和 loss
    lr0=0.01  # 初始学习率
)


if __name__ == "__main__":
    main()
