import os
import json
import glob
import shutil
import random
import os.path as osp

"""
数据集划分与 YOLO 数据集描述文件生成脚本

本脚本用于：
- 检查 `labels_dir`（标注目录）中是否已包含每张图片对应的 `.txt`（YOLO 格式）文件；
- 按图片列表随机划分数据集（支持两种模式）：
  - 2-way（train/val = 8:2），当用户输入 `n` 时启用；
  - 3-way（train/test/val = 7:2:1），当用户输入 `y` 或直接回车（默认）时启用；
- 将划分结果复制到 `dataset_output` 下的 `images/{train,val,test}` 和 `labels/{train,val,test}` 子目录；
- 根据 `classification_txt_path` 中的类别顺序生成 `dataset.yaml`，包含 `path`、`train`、`val`、可选 `test`、`nc`、`names` 字段，供 YOLO 训练使用。

主要配置（位于文件顶部，需根据项目调整）:
- `raw_data`：原始数据根目录（脚本默认使用相对路径，例如 `raw_data/tomato`）。
- `classification_txt_path`：包含类别名称的文本文件（每行一个类别），脚本会按此文件顺序生成 `names`。
- `images_dir`：图片目录（脚本会在此查找图片，支持常见扩展名和大小写）。
- `labels_dir`：标注目录，期望包含与图片同名的 `.txt` 文件（YOLO 格式）。
- `dataset_output`：划分后数据输出目录（包含 `images/` 和 `labels/` 子目录）。

运行流程：
1. 读取并加载配置（请确保 `classification_txt_path` 指向有效文件）。
2. 检查 `images_dir` 下的每张图片是否在 `labels_dir` 有对应 `.txt`，若缺失脚本会中止并打印缺失项。
3. 提示用户选择是否生成 test 集合（默认生成 3-way 划分）；按比例随机划分并复制对应图片与 `.txt`。
4. 在 `dataset_output` 中生成 `dataset.yaml`；打印结果路径。

示例运行：
    python 01_convert_labelme_to_yolo_seg.py

注意：
- 本脚本不再包含 JSON→TXT 的转换逻辑；如果你的标注是 JSON（Labelme/ISAT），请先使用相应转换工具生成 YOLO 格式的 `.txt`。
- 请确保 `classification_txt_path` 中的类别名称与标注文件中使用的类别一致，否则有些标注会被跳过。
"""

# =====================
# 配置区（已修改：支持图片与标注分开放在不同文件夹）
# 默认使用脚本同级目录下的 `raw_data/` 作为根目录，图片与标注分别放在
# `raw_datasets/{dataset_name}/images/` 和 `raw_datasets/{dataset_name}/labels/`。如需其他路径，请修改下面变量。
# =====================
dataset_process = input("处理增强后数据集还是原始数据集？\r\n输入 y 则处理增强后的数据集，输入 n 则处理原始数据集：").strip().lower()
if dataset_process == "y":
    candidate_aug = "tomato_augment"  # 增强后数据集名称
    # 如果增强数据集不存在，则回退为原始数据集
    if not os.path.isdir(f"raw_datasets/{candidate_aug}"):
        print(f"⚠️ 增强数据集不存在: raw_datasets/{candidate_aug}，将对原始数据集进行划分。")
        # 试图推断原始数据集名（去掉 `_augment` 后缀），否则使用默认 'tomato'
        base_name = candidate_aug.replace('_augment', '')
        if os.path.isdir(f"raw_datasets/{base_name}"):
            dataset_name = base_name
            print(f"使用原始数据集: {dataset_name}")
        else:
            dataset_name = 'tomato'
            print(f"未找到原始数据集，使用默认: {dataset_name}。请检查 raw_datasets/ 目录。")
    else:
        dataset_name = candidate_aug
else:
    dataset_name = "tomato"  # 数据集名称
    
raw_data = f"raw_datasets/{dataset_name}"  # 原始数据根目录
dataset_output = f"datasets/{dataset_name}"  # 划分后数据输出目录
classification_txt_path = rf"raw_datasets/{dataset_name}/labels/classification.txt"
with open(classification_txt_path, 'r', encoding='utf-8') as f:
    class_list = [line.strip() for line in f if line.strip()]
# 图片文件夹（修改为你实际的图片文件夹名，例如 'images' 或 'JPEGImages'）
images_dir = osp.join(raw_data, "images")
# 标注文件夹（包含 .json/.txt，例如 'labels'）
labels_dir = osp.join(raw_data, "labels")

random.seed(42)

# =====================
# 主流程
# =====================
def main():
    yolo_seg_splitter = YoloDatasetSplitter(dataset_name, images_dir, labels_dir, dataset_output, class_list)
    # -----------------------
    # 步骤 1：检查 TXT 是否完整
    # -----------------------
    yolo_seg_splitter.check_txt_files()

    # -----------------------
    # 步骤 2：数据划分（按图片列表划分并复制对应的 TXT）
    # -----------------------
    yolo_seg_splitter.dataset_split()
    # print("步骤2：准备划分数据集...")

    # -----------------------
    # 步骤 3：生成 dataset YAML 文件
    # -----------------------
    yolo_seg_splitter.generate_yaml()


class YoloDatasetSplitter:
    def __init__(self, dataset_name, images_dir, labels_dir, dataset_output, class_list):
        self.dataset_name = dataset_name
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.dataset_output = dataset_output
        self.class_list = class_list
        self.pic_formats = [".jpeg", ".JPEG", ".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP", ".tif", ".TIF", ".tiff", ".TIFF", ".webp", ".WEBP"]
        self.image_files = []
        
        self.is_testDataset_required = False
        
        for pic_format in self.pic_formats:
            self.image_files = self.image_files + glob.glob(osp.join(self.images_dir, f"*{pic_format}"))

    def make_yolo_dirs(self):
        """创建 YOLO 所需目录"""
        dirs = ["images/train", "images/val", "labels/train", "labels/val"]
        if self.is_testDataset_required:
            dirs.extend(["images/test", "labels/test"])
        for d in dirs:
            path = osp.join(self.dataset_output, d)
            if not osp.exists(path):
                os.makedirs(path)
        print("✅ 目录检查完成")


    def find_image(self, base):
        """在 `images_dir` 中查找图片，支持多种后缀（大小写不敏感）。

        优先返回第一个匹配的常见图片文件。
        """
        # 尝试精确后缀匹配（常见小写后缀）
        for ext in self.pic_formats:
            img = osp.join(self.images_dir, base + ext)
            if osp.exists(img):
                return img

        # 如果没找到，用 glob 匹配任意扩展并检查扩展是否为图片格式（大小写兼容）
        candidates = glob.glob(osp.join(self.images_dir, base + ".*"))
        for c in candidates:
            ext = osp.splitext(c)[1].lower()
            if ext in self.pic_formats:
                return c

        return None
    
    def copy_split(self, basenames, subset_name):
        for base in basenames:
            img_path = self.find_image(base)
            if img_path is None:
                print(f"⚠ 找不到图片：{base}，已跳过")
                continue
            dst_img = osp.join(self.dataset_output, "images", subset_name, osp.basename(img_path))
            shutil.copy(img_path, dst_img)

            src_txt = osp.join(self.labels_dir, base + ".txt")
            dst_txt = osp.join(self.dataset_output, "labels", subset_name, base + ".txt")
            if osp.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
            else:
                print(f"⚠ 未找到标注 TXT：{base}.txt（在 labels_dir 中），已跳过）")
    
    def check_txt_files(self):
        """检查每张图片是否都有对应的 TXT 文件"""
        for img in self.image_files:
            base = osp.splitext(osp.basename(img))[0]
            txt_path = osp.join(self.labels_dir, base + ".txt")
            if not osp.exists(txt_path):
                print(f"❌ 缺少 TXT 文件（在标注文件夹中）：{base}.txt")
                return False
        return True
    
    def dataset_split(self):
        # 获取所有图片 basenames
        bases = [osp.splitext(osp.basename(p))[0] for p in self.image_files]
        if not bases:
            print("❌ 未在 images_dir 中找到任何图片，无法划分数据集。")
            return

        # 用户选择是否包含 test 集合
        opt_test = input("是否划分 test 集合？\r\n回车或输入 y 则划分 train:test:val=7:2:1，输入 n 则只划分 train:val=8:2：(y/n, default y)：").strip().lower()
        self.is_testDataset_required = (opt_test != "n")
        self.make_yolo_dirs()

        random.shuffle(bases)
        if self.is_testDataset_required:  
            n = len(bases)
            n_train = int(n * 0.7)
            n_test = int(n * 0.2)
            train_bases = bases[:n_train]
            test_bases = bases[n_train:n_train + n_test]
            val_bases = bases[n_train + n_test:]
            print(f"➡ 样本总数: {n}，训练: {len(train_bases)}，测试: {len(test_bases)}，验证: {len(val_bases)}")
        else:
            n = len(bases)
            n_train = int(n * 0.8)
            train_bases = bases[:n_train]
            val_bases = bases[n_train:]
            test_bases = []
            print(f"➡ 样本总数: {n}，训练: {len(train_bases)}，验证: {len(val_bases)} (无测试集)")

        # 执行复制
        self.copy_split(train_bases, "train")
        if self.is_testDataset_required:
            self.copy_split(test_bases, "test")
        self.copy_split(val_bases, "val")

        print(f"🎉 数据划分完成！所有数据已存入 {self.dataset_output}/ 目录")
        
    def generate_yaml(self):
        # 生成 dataset YAML 文件
        yaml_path = osp.join(self.dataset_output, f"{self.dataset_name}.yaml")
        dataset_path = self.dataset_name.replace('\\', '/')
        lines = []
        lines.append(f'# YOLO 数据集描述文件，仅适配 Ultralytics')
        lines.append(f'path: "{dataset_path}"')
        lines.append("")
        lines.append("train: images/train")
        lines.append("val: images/val")
        if self.is_testDataset_required:
            lines.append("test: images/test")
        lines.append("")
        lines.append(f"nc: {len(self.class_list)}")
        lines.append("names:")
        for i, name in enumerate(self.class_list):
            lines.append(f"  {i}: {name}")
        lines.append("")
        with open(yaml_path, 'w', encoding='utf-8') as yf:
            yf.write('\n'.join(lines))

        print(f"✅ 已生成 YAML 文件：{yaml_path}")


if __name__ == "__main__":
    main()
