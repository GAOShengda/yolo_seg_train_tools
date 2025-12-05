"""
数据集增强（Orchestrator）

DATASET_NAME: 数据集名称（文件夹名）

在文件顶部配置加载要使用的增强工具字典与参数。示例：

TOOLS = {
    'blur': {'enabled': True, 'radius': 10, 'suffix': '_blur'},
    'color_jitter': {'enabled': True, 'variants': [ ... ]},
}

工具字典：若某工具 'enabled' 为 True 则会执行
字段说明（中文注释）：
- blur:
    - enabled: 是否启用该增强工具（True/False）
    - radius: 高斯模糊半径（数值，越大越模糊）
    - suffix: 输出文件名后缀（例如 '_blur'，会生成 IMG_0001_blur.jpg）
    - replace_imagedata: 如果原 JSON 中包含 `imageData`（base64），是否用增强后的图片的 base64 替换（True/False）
- color_jitter:
    - enabled: 是否启用该增强工具（True/False）
    - variants: 一个变体列表，每个变体为字典，描述要生成的具体色彩变换；示例字段：
        - suffix: 该变体输出文件名后缀（例如 '_b0.9_c0.95_s0.9'）
        - brightness: 亮度乘数（1.0 不变，0.9 稍暗，1.1 稍亮）
        - contrast: 对比度乘数（1.0 不变）
        - saturation: 饱和度乘数（1.0 不变）
        - hue: 色相偏移，单位为度，取值范围约为 -180..180（0 不变）
    - replace_imagedata: 同上，是否替换 JSON 中的 imageData（True/False）


运行：编辑顶部配置后直接运行 `python3 数据集增强.py`（在 dataset 根目录）
"""
import sys
import shutil
from pathlib import Path

from tools.dataset_augment import BlurAugment, ColorJitterAugment

# -------------------- 在这里编辑要使用的工具与参数 --------------------
DATASET_ROOT = 'raw_datasets'
DATASET_NAME = 'tomato'


TOOLS = {
    'blur': {
        'enabled': True,
        'radius': 10,
        # 按比例随机抽取要增强的样本（0.0-1.0），例如 0.2 表示抽取 20% 的图片
        'sample_ratio': 0.2,
        'replace_imagedata': True,
    },
    'color_jitter': {
        'enabled': True,
        'variants': [
            {'brightness': 0.9, 'contrast': 0.95, 'saturation': 0.9, 'hue': 0},
            {'brightness': 1.1, 'contrast': 1.05, 'saturation': 1.1, 'hue': 0},
            {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0, 'hue': -10},
            {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0, 'hue': 10},
        ],
        # 每个工具单独随机抽样，保证不同工具使用不同随机子集
        'sample_ratio': 0.2,
        'replace_imagedata': True,
    }
}
# -------------------------------------------------------------------


def error(msg: str):
    # red text
    print(f"\033[31mERROR: {msg}\033[0m", file=sys.stderr)


def check_dataset(root: Path, name: str) -> bool:
    ds = root / name
    if not ds.exists():
        error(f'dataset folder not found: {ds}')
        return False
    imgs = ds / 'images'
    labels = ds / 'labels'
    if not (imgs.exists() or labels.exists()):
        error(f'No image folder found under {ds} (expected `img` or `image`)')
        return False
    if not labels.exists():
        error(f'No labels folder found under {ds} (expected `labels`)')
        return False
    return True


def main():
    root = Path(DATASET_ROOT).resolve()
    if not check_dataset(root, DATASET_NAME):
        sys.exit(1)

    # create augmented dataset folder (copy originals there)
    def make_unique_ds_name(base: Path, name: str) -> str:
        candidate = f"{name}_augment"
        i = 1
        while (base / candidate).exists():
            candidate = f"{name}_augment_{i}"
            i += 1
        return candidate

    new_ds_name = make_unique_ds_name(root, DATASET_NAME)
    new_ds_path = root / new_ds_name
    # copy original dataset directories (`img` and `label`) into new dataset
    orig_ds = root / DATASET_NAME
    try:
        new_ds_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # unlikely due to make_unique_ds_name, but handle defensively
        pass
    # copy img and label
    for sub in ('images', 'labels'):
        src = orig_ds / sub
        dst = new_ds_path / sub
        if src.exists():
            try:
                shutil.copytree(src, dst)
            except FileExistsError:
                # merge by copying files individually
                for p in src.iterdir():
                    if p.is_file():
                        shutil.copy2(p, dst / p.name)
            except Exception as e:
                print(f'Failed to copy {src} -> {dst}: {e}', file=sys.stderr)
        else:
            # create empty dirs if missing
            dst.mkdir(parents=True, exist_ok=True)

    print(f'Created augmented dataset: {new_ds_path}')

    # run selected tools
    if TOOLS.get('blur', {}).get('enabled'):
        cfg = TOOLS['blur']
        # pass the whole cfg so augmenter can pick needed keys
        aug = BlurAugment(cfg)
        # run on the copied dataset so originals are preserved; write into same dataset
        print(f'Running blur on dataset {new_ds_name} -> radius={cfg.get("radius")}')
        aug.run(dataset_root=DATASET_ROOT, dataset_name=new_ds_name, out_dir=new_ds_name)

    if TOOLS.get('color_jitter', {}).get('enabled'):
        cfg = TOOLS['color_jitter']
        # pass the whole cfg so augmenter can pick needed keys
        aug = ColorJitterAugment(cfg)
        print(f'Running color_jitter on dataset {new_ds_name} -> variants={len(cfg.get("variants", []))}')
        aug.run(dataset_root=DATASET_ROOT, dataset_name=new_ds_name, out_dir=new_ds_name)

    print('All selected augmentations finished.')


if __name__ == '__main__':
    main()
