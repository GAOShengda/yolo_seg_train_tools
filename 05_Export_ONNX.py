from ultralytics import YOLO

# 指定模型权重的完整路径
model = YOLO(r"F:\zywXM\PyCharmPro\yolov8-seg\runs\train\potato\weights\best.pt")

# 导出为 ONNX 生成在model同级目录
model.export(format="onnx")
