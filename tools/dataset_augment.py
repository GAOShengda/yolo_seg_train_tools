"""Class-based blur augmenter

提供 BlurAugment 类：通过配置在代码顶部或由外部传入参数执行模糊增强并输出图片/JSON/TXT
"""
import os
import json
import base64
import shutil
import sys
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import subprocess
import random
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
    try:
        # try to install tqdm using the same python interpreter and TUNA mirror
        cmd = [sys.executable, '-m', 'pip', 'install', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple', 'tqdm']
        subprocess.run(cmd, check=False)
        from tqdm import tqdm  # try import again
    except Exception:
        tqdm = None

SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def pil_format_from_ext(ext: str):
    ext = ext.lower()
    if ext in ['.jpg', '.jpeg']:
        return 'JPEG'
    if ext in ['.png']:
        return 'PNG'
    if ext in ['.tif', '.tiff']:
        return 'TIFF'
    if ext in ['.bmp']:
        return 'BMP'
    return 'PNG'


def image_to_base64(img: Image.Image, fmt: str):
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def select_samples_by_json_files(dataset_root='.', dataset_name='tomato', sample_ratio: float | None = None, sample_count: int | None = None, seed: int | None = None) -> list | None:
    """根据比例或数量在 labels 目录中随机选择样本。
    要求目录下必须存在 JSON 文件。

    返回值：
    - None: 表示使用全部样本
    - []: 表示选择 0 个样本
    - list[Path]: 选中的 json 文件路径列表
    """
    if seed is not None:
        random.seed(seed)
    root = Path(dataset_root).resolve()
    labels_dir = root / dataset_name / 'labels'
    all_jsons = sorted(labels_dir.glob('*.json'))
    total = len(all_jsons)
    if sample_count is not None:
        if sample_count <= 0:
            return []
        if sample_count >= total:
            return None
        return random.sample(all_jsons, min(sample_count, total))
    if sample_ratio is not None:
        r = float(sample_ratio)
        if r <= 0:
            return []
        if r >= 1.0:
            return None
        k = max(1, int(total * r))
        return random.sample(all_jsons, k) if all_jsons else []
    return None


class BlurAugment:
    """高斯模糊增强器

    参数化：radius, suffix, replace_imagedata
    方法：run(dataset_root, dataset_name, out_dir)
    """

    def __init__(self, cfg: dict | None = None, **kwargs):
        """构造函数支持两种写法：

        - 关键字展开： `BlurAugment(**cfg)`
        - 传入 dict： `BlurAugment(cfg)`（更直观）

        参数解析优先级：`cfg` 中的键 -> 再被 `kwargs` 覆盖（如果同时提供）。

        支持的键：`radius`, `suffix`, `replace_imagedata`, `sample_ratio`, `sample_count`, `sample_seed`
        """
        merged = {}
        if isinstance(cfg, dict):
            merged.update(cfg)
        merged.update(kwargs)

        self.radius = merged.get('radius', 5.0)
        self.suffix = merged.get('suffix', '_blur')
        self.replace_imagedata = merged.get('replace_imagedata', True)
        self.sample_ratio = merged.get('sample_ratio')
        self.sample_count = merged.get('sample_count')
        self.sample_seed = merged.get('sample_seed')

    def find_image_file(self, img_dir: Path, base_name: str):
        for ext in SUPPORTED_EXTS:
            p = img_dir / (base_name + ext)
            if p.exists():
                return p
        for p in img_dir.iterdir() if img_dir.exists() else []:
            if p.is_file() and p.stem.lower() == base_name.lower() and p.suffix.lower() in SUPPORTED_EXTS:
                return p
        return None

    def update_json_image_info(self, json_data: dict, new_filename: str, b64_data: str | None):
        if 'imagePath' in json_data:
            json_data['imagePath'] = new_filename
        if 'imageFilename' in json_data:
            json_data['imageFilename'] = new_filename
        if 'imageData' in json_data and b64_data is not None and self.replace_imagedata:
            json_data['imageData'] = b64_data

    def run(self, dataset_root='.', dataset_name='tomato', out_dir='blur', sample_list: list | None = None):
        root = Path(dataset_root).resolve()
        ds = root / dataset_name
        labels_dir = ds / 'labels'
        img_dir = ds / 'images'

        out_base = root / out_dir
        out_img_dir = out_base / 'images'
        out_labels_dir = out_base / 'labels'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        all_json_files = sorted(labels_dir.glob('*.json'))
        # if caller didn't provide explicit sample_list, use stored sampling params
        if sample_list is None:
            sample_list = select_samples_by_json_files(dataset_root=dataset_root, dataset_name=dataset_name, sample_ratio=self.sample_ratio, sample_count=self.sample_count, seed=self.sample_seed)

        # If caller provided a sample_list (list of Paths or filenames/stems), map them to json paths
        if sample_list:
            mapped = []
            for s in sample_list:
                p = Path(s)
                # if absolute path and exists, accept
                if p.is_absolute() and p.exists():
                    mapped.append(p)
                    continue
                # try relative to labels_dir
                cand = labels_dir / p
                if cand.exists():
                    mapped.append(cand)
                    continue
                # try stem -> add .json
                cand2 = labels_dir / (p.stem + '.json')
                if cand2.exists():
                    mapped.append(cand2)
                    continue
            # fallback to all if mapping failed
            json_files = mapped if mapped else all_json_files
        else:
            json_files = all_json_files
        total = 0
        skipped = 0
        iterator = tqdm(json_files, desc=f'Blur (samples={len(json_files)})') if tqdm else json_files
        for jpath in iterator:
            total += 1
            try:
                with open(jpath, 'r', encoding='utf-8') as f:
                    j = json.load(f)
            except Exception as e:
                skipped += 1
                if tqdm:
                    tqdm.write(f'Failed to read {jpath}: {e}')
                else:
                    print(f'Failed to read {jpath}: {e}', file=sys.stderr)
                continue

            base_name = jpath.stem
            candidate = None
            for key in ('imagePath', 'imageFilename', 'image_name'):
                if key in j and isinstance(j[key], str) and j[key].strip():
                    candidate = j[key].strip()
                    break

            img_path = None
            if candidate:
                cand_name = os.path.basename(candidate)
                cand_stem, cand_ext = os.path.splitext(cand_name)
                if cand_ext:
                    p = img_dir / cand_name
                    if p.exists():
                        img_path = p
                else:
                    img_path = self.find_image_file(img_dir, cand_name)

            if img_path is None:
                img_path = self.find_image_file(img_dir, base_name)

            if img_path is None:
                skipped += 1
                continue

            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                skipped += 1
                if tqdm:
                    tqdm.write(f'Failed to open image {img_path}: {e}')
                else:
                    print(f'Failed to open image {img_path}: {e}', file=sys.stderr)
                continue

            blurred = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
            ext = img_path.suffix or '.jpg'
            # filename format: <suffix_without_underscore>_<original_stem><ext>
            suf = self.suffix.lstrip('_')
            out_img_name = f'{suf}_{img_path.stem}{ext}'
            out_img_path = out_img_dir / out_img_name
            try:
                pil_fmt = pil_format_from_ext(ext)
                blurred.save(out_img_path, format=pil_fmt)
            except Exception as e:
                skipped += 1
                if tqdm:
                    tqdm.write(f'Failed to save blurred image {out_img_path}: {e}')
                else:
                    print(f'Failed to save blurred image {out_img_path}: {e}', file=sys.stderr)
                continue

            b64 = None
            if 'imageData' in j and self.replace_imagedata:
                pil_fmt = pil_format_from_ext(ext)
                b64 = image_to_base64(blurred, pil_fmt)

            self.update_json_image_info(j, out_img_name, b64)

            out_json_name = f'{suf}_{base_name}.json'
            out_json_path = out_labels_dir / out_json_name
            try:
                with open(out_json_path, 'w', encoding='utf-8') as f:
                    json.dump(j, f, ensure_ascii=False, indent=2)
            except Exception as e:
                skipped += 1
                if tqdm:
                    tqdm.write(f'Failed to write json {out_json_path}: {e}')
                else:
                    print(f'Failed to write json {out_json_path}: {e}', file=sys.stderr)
                continue

            # copy txt if exists
            txt_src = labels_dir / f'{base_name}.txt'
            if txt_src.exists():
                txt_dst = out_labels_dir / f'{base_name}{self.suffix}.txt'
                try:
                    shutil.copy2(txt_src, txt_dst)
                    txt_status = 'copied'
                except Exception as e:
                    txt_status = 'error'
                    if tqdm:
                        tqdm.write(f'Failed to copy txt {txt_src} -> {txt_dst}: {e}')
            else:
                txt_status = 'none'

            # update progress postfix if available
            if tqdm and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({'last': out_img_name, 'txt': txt_status})

        if tqdm:
            tqdm.write(f'Summary blur: total={total} skipped={skipped} saved_to={out_base}')
        else:
            print(f'Summary blur: total={total} skipped={skipped} saved_to={out_base}')


class ColorJitterAugment:
    """色彩抖动增强器

    VARIANTS: list of dicts with keys: suffix, brightness, contrast, saturation, hue
    """

    def __init__(self, cfg: dict | None = None, **kwargs):
        """构造函数支持两种写法：

        - 关键字展开： `ColorJitterAugment(**cfg)`
        - 传入 dict： `ColorJitterAugment(cfg)`

        参数解析优先级：`cfg` 中的键 -> 再被 `kwargs` 覆盖（如果同时提供）。

        支持的键：`variants`, `replace_imagedata`, `continue_on_hue_error`, `sample_ratio`, `sample_count`, `sample_seed`
        """
        merged = {}
        if isinstance(cfg, dict):
            merged.update(cfg)
        merged.update(kwargs)

        self.variants = merged.get('variants', [])
        self.replace_imagedata = merged.get('replace_imagedata', True)
        self.continue_on_hue_error = merged.get('continue_on_hue_error', True)
        # sampling config
        self.sample_ratio = merged.get('sample_ratio')
        self.sample_count = merged.get('sample_count')
        self.sample_seed = merged.get('sample_seed')

    def find_image_file(self, img_dir: Path, base_name: str):
        for ext in SUPPORTED_EXTS:
            p = img_dir / (base_name + ext)
            if p.exists():
                return p
        for p in img_dir.iterdir() if img_dir.exists() else []:
            if p.is_file() and p.stem.lower() == base_name.lower() and p.suffix.lower() in SUPPORTED_EXTS:
                return p
        return None

    def shift_hue(self, img: Image.Image, deg: float):
        if deg == 0:
            return img
        try:
            import numpy as np
        except Exception:
            raise RuntimeError('numpy is required for hue shift')
        hsv = img.convert('HSV')
        arr = np.array(hsv, dtype=np.int16)
        shift = int(deg / 360.0 * 255.0)
        arr[..., 0] = (arr[..., 0] + shift) % 256
        arr = arr.astype('uint8')
        from PIL import Image as PILImage
        return PILImage.fromarray(arr, mode='HSV').convert('RGB')

    def apply_variant(self, img: Image.Image, variant: dict):
        out = img
        if variant.get('brightness', 1.0) != 1.0:
            out = ImageEnhance.Brightness(out).enhance(variant['brightness'])
        if variant.get('contrast', 1.0) != 1.0:
            out = ImageEnhance.Contrast(out).enhance(variant['contrast'])
        if variant.get('saturation', 1.0) != 1.0:
            out = ImageEnhance.Color(out).enhance(variant['saturation'])
        hue = variant.get('hue', 0)
        if hue and hue != 0:
            out = self.shift_hue(out, hue)
        return out

    def make_suffix_from_params(self, var: dict) -> str:
        """自动根据 variant 参数生成后缀，避免手动维护 suffix。

        规则：
        - 对于 brightness/contrast/saturation 使用两位小数格式并去掉不必要的零
        - hue 以整数形式加入（若为 0 则不加入）
        - 如所有参数均为默认值，则返回 '_identity'
        """
        b = var.get('brightness', 1.0)
        c = var.get('contrast', 1.0)
        s = var.get('saturation', 1.0)
        h = var.get('hue', 0)

        def fmt(x):
            if isinstance(x, int):
                return str(x)
            t = '{:.2f}'.format(x).rstrip('0').rstrip('.')
            return t

        parts = []
        if b != 1.0:
            parts.append('b' + fmt(b))
        if c != 1.0:
            parts.append('c' + fmt(c))
        if s != 1.0:
            parts.append('s' + fmt(s))
        if h != 0:
            parts.append('h' + fmt(int(h)))
        if not parts:
            return '_identity'
        return '_' + '_'.join(parts)

    def update_json_image_info(self, json_data: dict, new_filename: str, b64_data: str | None):
        if 'imagePath' in json_data:
            json_data['imagePath'] = new_filename
        if 'imageFilename' in json_data:
            json_data['imageFilename'] = new_filename
        if 'imageData' in json_data and b64_data is not None and self.replace_imagedata:
            json_data['imageData'] = b64_data

    def run(self, dataset_root='.', dataset_name='tomato', out_dir='color_jitter', sample_list: list | None = None):
        root = Path(dataset_root).resolve()
        ds = root / dataset_name
        labels_dir = ds / 'labels'
        img_dir = ds / 'images'

        out_base = root / out_dir
        out_img_dir = out_base / 'images'
        out_labels_dir = out_base / 'labels'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        all_json_files = sorted(labels_dir.glob('*.json'))
        # if caller didn't provide sample_list, use stored sampling params
        if sample_list is None:
            sample_list = select_samples_by_json_files(dataset_root=dataset_root, dataset_name=dataset_name, sample_ratio=self.sample_ratio, sample_count=self.sample_count, seed=self.sample_seed)

        # map provided sample_list (stems, filenames or Paths) to json paths
        if sample_list:
            mapped = []
            for s in sample_list:
                p = Path(s)
                if p.is_absolute() and p.exists():
                    mapped.append(p)
                    continue
                cand = labels_dir / p
                if cand.exists():
                    mapped.append(cand)
                    continue
                cand2 = labels_dir / (p.stem + '.json')
                if cand2.exists():
                    mapped.append(cand2)
                    continue
            json_files = mapped if mapped else all_json_files
        else:
            json_files = all_json_files
        total = 0
        skipped = 0
        iterator = tqdm(json_files, desc=f'ColorJitter (samples={len(json_files)})') if tqdm else json_files
        # prepare RNG: if sample_seed provided, use it for reproducibility
        rng = random.Random(self.sample_seed) if self.sample_seed is not None else random

        for jpath in iterator:
            total += 1
            try:
                with open(jpath, 'r', encoding='utf-8') as f:
                    j = json.load(f)
            except Exception as e:
                skipped += 1
                if tqdm:
                    tqdm.write(f'Failed to read {jpath}: {e}')
                else:
                    print(f'Failed to read {jpath}: {e}', file=sys.stderr)
                continue

            base_name = jpath.stem
            img_path = self.find_image_file(img_dir, base_name)
            if img_path is None:
                skipped += 1
                continue

            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                skipped += 1
                if tqdm:
                    tqdm.write(f'Failed to open image {img_path}: {e}')
                else:
                    print(f'Failed to open image {img_path}: {e}', file=sys.stderr)
                continue

            # For each sampled image, pick one variant at random (not apply all variants)
            if not self.variants:
                # nothing to do
                continue
            var = rng.choice(self.variants)
            suffix = var.get('suffix') or self.make_suffix_from_params(var)
            # place parameter part before original name, remove leading underscore
            param_part = suffix.lstrip('_')
            out_img_name = f'{param_part}_{img_path.stem}{img_path.suffix}'
            out_img_path = out_img_dir / out_img_name
            try:
                enhanced = None
                try:
                    enhanced = self.apply_variant(img, var)
                except Exception as e:
                    if 'hue' in var and 'numpy' in str(e).lower() and self.continue_on_hue_error:
                        vv = var.copy()
                        vv['hue'] = 0
                        enhanced = self.apply_variant(img, vv)
                    else:
                        raise
                pil_fmt = pil_format_from_ext(img_path.suffix)
                enhanced.save(out_img_path, format=pil_fmt)
            except Exception as e:
                skipped += 1
                if tqdm:
                    tqdm.write(f'Failed to apply/save variant {suffix} for {img_path.name}: {e}')
                else:
                    print(f'Failed to apply/save variant {suffix} for {img_path.name}: {e}', file=sys.stderr)
                continue

            b64 = None
            if 'imageData' in j and self.replace_imagedata:
                pil_fmt = pil_format_from_ext(img_path.suffix)
                b64 = image_to_base64(enhanced, pil_fmt)

            j_new = dict(j)
            self.update_json_image_info(j_new, out_img_name, b64)

            out_json_name = f'{param_part}_{base_name}.json'
            out_json_path = out_labels_dir / out_json_name
            try:
                with open(out_json_path, 'w', encoding='utf-8') as f:
                    json.dump(j_new, f, ensure_ascii=False, indent=2)
            except Exception as e:
                skipped += 1
                if tqdm:
                    tqdm.write(f'Failed to write json {out_json_path}: {e}')
                else:
                    print(f'Failed to write json {out_json_path}: {e}', file=sys.stderr)
                continue

            # copy txt if exists
            txt_src = labels_dir / f'{base_name}.txt'
            if txt_src.exists():
                txt_dst = out_labels_dir / f'{param_part}_{base_name}.txt'
                try:
                    shutil.copy2(txt_src, txt_dst)
                    txt_status = 'copied'
                except Exception as e:
                    txt_status = 'error'
                    if tqdm:
                        tqdm.write(f'Failed to copy txt {txt_src} -> {txt_dst}: {e}')
            else:
                txt_status = 'none'

            if tqdm and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({'last': out_img_name, 'txt': txt_status})

        print(f'Summary color_jitter: total={total} skipped={skipped} saved_to={out_base}')
