import os
import glob
import shutil
import datetime

"""
使用ISAT标注时使用内部工具转化，打开转化后的text查看类别是否从0开始，若从1开始则需要执行该脚本
如果没有使用ISAT则不需要执行该脚本

类别ID转换说明：
-----------------
该脚本用于将分割标签文件中的类别ID按照预定义的映射规则进行转换。

功能：
- 将类别ID从一个值转换为另一个值（例如：类别1 -> 类别0，类别2 -> 类别1）。
- 支持批量处理指定目录下的所有标签文件。
- 可选备份功能，确保原始文件不会丢失。

示例：
    假设标签文件内容如下：
    文件1.txt：
    2 0.5 0.5 0.2 0.2
    5 0.3 0.3 0.1 0.1
    8 0.4 0.4 0.2 0.2

    如果类别映射规则为：
    class_remapping = {
        2: 0,  # 旧类别2 -> 新类别0
        5: 1,  # 旧类别5 -> 新类别1
        8: 2   # 旧类别8 -> 新类别2
    }

    转换后文件内容：
    文件1.txt：
    0 0.5 0.5 0.2 0.2
    1 0.3 0.3 0.1 0.1
    2 0.4 0.4 0.2 0.2

注意：
- 类别2、5、8被重新映射为0、1、2。
- 如果需要动态生成类别映射（例如：将所有用到的类别重新整理为0, 1, 2...），需要修改脚本逻辑。
- 请确保在运行脚本前备份重要数据。

"""
DATASET_NAME = "tomato"  # 数据集名称

def main():
    """主函数"""
    classification_txt_path = rf"raw_data/{DATASET_NAME}/labels/classification.txt"
    class_remapping = {
            2: 0,  # 类别1 -> 类别0
            3: 1  # 类别2 -> 类别1
        }

    # 配置路径
    labels_dir = rf"raw_data/{DATASET_NAME}/labels"
    backup_dir = rf"raw_data/{DATASET_NAME}/labels_backup"

    print("分割标签类别ID转换脚本")
    print("=" * 80)
    print(f"标签目录: {labels_dir}")
    print(f"备份目录: {backup_dir}")
    print("=" * 80)

    # 检查输入目录是否存在
    if not os.path.exists(labels_dir):
        print(f"错误: 标签目录不存在: {labels_dir}")
        return

    print("建议先备份重要数据。")
    
    confirm = input("\n数据标注工具是否为ISAT (ISAT 需要进行转换，输入 'y' 确认): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return

    try:
        # 创建转换器并运行
        converter = ClassIDConverter(
            labels_dir=labels_dir,
            backup_dir=backup_dir,
            class_remapping=class_remapping
        )
        proceed = converter.backup_and_filter_classification(classification_txt_path)

        # 如果备份/分类处理返回 False，表示y选择不继续或发生错误，终止程序
        if not proceed:
            return

        # 运行转换
        converter.run_conversion()

    except Exception as e:
        print(f"运行失败: {e}")
        import traceback
        traceback.print_exc()
        
        
class ClassIDConverter:
    def __init__(self, labels_dir, backup_dir=None, class_remapping:dict=None):
        """
        初始化转换器

        Args:
            labels_dir: 分割标签文件夹路径
            backup_dir: 备份文件夹路径（可选）
        """
        self.labels_dir = labels_dir
        self.backup_dir = backup_dir
        self.class_remapping = class_remapping
        self.full_backup_done = False
        
        # 转换统计
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'total_annotations': 0,
            'converted_annotations': 0,
            'dropped_annotations': 0
        }
        
    def backup_and_filter_classification(self, classification_txt_path):
        """
        读取 classification.txt，展示原始类别与新的类别映射，要求用户确认。
        若用户确认，则备份 labels_dir（若指定），并将修改后的分类写回 classification.txt。
        返回 True 表示可以继续后续转换，返回 False 则中止流程。
        """
        try:
            if not os.path.exists(classification_txt_path):
                red = "\033[31m"
                reset = "\033[0m"
                print(f"{red}错误: classification.txt 不存在: {classification_txt_path}\n请检查 classification.txt 文件！！！{reset}")
                return False

            # 记录 classification 文件路径，后续遍历时跳过该文件，避免误修改
            self.classification_path = os.path.abspath(classification_txt_path)

            # 先读取原始 classification.txt 内容并准备展示
            with open(classification_txt_path, 'r', encoding='utf-8') as f:
                lines = [l.rstrip('\n') for l in f.readlines()]

            # 检查文件是否为空或仅包含空行
            if not any((ln.strip() for ln in lines)):
                red = "\033[31m"
                reset = "\033[0m"
                print(f"{red}错误: classification.txt 文件为空: {classification_txt_path}\n请检查 classification.txt 文件！！！{reset}")
                return False

            kept_lines = []
            kept_map = {}
            deleted = []
            for idx, line in enumerate(lines):
                name = line.strip()
                if not name or name.startswith('__background__'):
                    deleted.append((idx, name))
                    continue
                if idx in self.class_remapping:
                    kept_lines.append(name)
                    kept_map[idx] = name
                else:
                    deleted.append((idx, name))

            # 打印原始类别 id -> 名称
            print("\n原始 classification.txt 中的类别 (index: name):")
            for idx, name in enumerate(lines):
                print(f"  {idx}: {name}")

            # 生成并打印将要写入的新的分类列表
            max_new = max(self.class_remapping.values())
            new_names = [None] * (max_new + 1)
            for old_idx, new_idx in self.class_remapping.items():
                if old_idx < len(lines):
                    new_names[new_idx] = lines[old_idx].strip()
            final_names = [n for n in new_names if n]

            print("\n修改后的 classification.txt 内容为:")
            for new_idx, name in enumerate(final_names):
                print(f"  {new_idx}: {name}")

            # 如果修改后的内容为空，报错并终止流程
            if not final_names:
                red = "\033[31m"
                reset = "\033[0m"
                print(f"{red}错误: 修改后的类别映射为空，请检查classification.txt 或类别映射class_remapping是否存在问题！{reset}")
                return False

            # 询问是否确认类别名称修改
            try:
                confirm_names = input("\n确认要将 classification.txt 修改为以上内容吗？(y/n): ")
            except Exception:
                confirm_names = 'n'
                
            if confirm_names.lower() != 'y':
                red = "\033[31m"
                reset = "\033[0m"
                print(f"{red}类别名称修改未确认，程序结束。{reset}")
                return False

            if self.backup_dir:
                try:
                    if os.path.exists(self.backup_dir):
                        red = "\033[31m"
                        reset = "\033[0m"
                        try:
                            ans = input(f"{red}警告: 备份目录 '{self.backup_dir}' 已存在。是否覆盖并重新备份？(y=覆盖, n=跳过并退出): {reset}")
                        except Exception:
                            ans = 'n'

                        if ans.lower() == 'y':
                            try:
                                shutil.rmtree(self.backup_dir)
                                print(f"已删除旧备份目录: {self.backup_dir}，开始重新备份...")
                            except Exception as e:
                                print(f"警告: 删除旧备份目录失败: {e}")
                                print("取消操作。请手动检查备份目录权限或手动移除旧备份后重试。")
                                return False
                        else:
                            red = "\033[31m"
                            reset = "\033[0m"
                            print(f"{red}已存在旧备份，请检查是否符合要求。不进行 txt 标签转换，程序结束。{reset}")
                            return False

                    try:
                        shutil.copytree(self.labels_dir, self.backup_dir)
                    except Exception:
                        os.makedirs(self.backup_dir, exist_ok=True)
                        for item in os.listdir(self.labels_dir):
                            src_item = os.path.join(self.labels_dir, item)
                            dest_item = os.path.join(self.backup_dir, item)
                            if os.path.isdir(src_item):
                                if os.path.exists(dest_item):
                                    continue
                                shutil.copytree(src_item, dest_item)
                            else:
                                shutil.copy2(src_item, dest_item)

                    self.full_backup_done = True
                    print(f"已备份整个标签目录到: {self.backup_dir}")
                except Exception as e:
                    print(f"备份整个标签目录失败: {e}")
                    return False

            # 覆写 classification.txt
            try:
                with open(classification_txt_path, 'w', encoding='utf-8') as f:
                    for name in final_names:
                        f.write(name + '\n')
                        print(f"  写入新类别: {name}")
                print("classification.txt 已根据映射更新并保存在 labels 目录中。\n")
            except Exception as e:
                print(f"写入 classification.txt 失败: {e}")
                return False

            return True
        except Exception as e:
            print(f"处理 classification.txt 时出错: {e}")
            return False

    def create_backup(self):
        """创建备份文件夹"""
        if not self.backup_dir:
            return True

        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            print(f"✓ 备份目录已创建: {self.backup_dir}")
            return True
        except Exception as e:
            print(f"✗ 创建备份目录失败: {e}")
            return False

    def backup_file(self, file_path):
        """备份单个文件"""
        # 如果已经做过整个目录的备份（full_backup_done），则不要再往 backup_dir 写入文件，避免修改备份
        if self.backup_dir and self.full_backup_done:
            return True

        if not self.backup_dir:
            return True

        try:
            filename = os.path.basename(file_path)
            backup_path = os.path.join(self.backup_dir, filename)
            shutil.copy2(file_path, backup_path)
            return True
        except Exception as e:
            print(f"    警告: 备份文件 {filename} 失败: {e}")
            return False

    def convert_single_file(self, file_path):
        """
        转换单个标签文件

        Args:
            file_path: 标签文件路径

        Returns:
            bool: 是否转换成功
        """
        try:
            filename = os.path.basename(file_path)

            # 创建备份
            if not self.backup_file(file_path):
                print(f"  ✗ 备份失败，跳过文件: {filename}")
                return False

            # 读取原文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 转换类别ID并同时删除不在映射表中的类别（合并过滤与转换）
            converted_lines = []
            file_annotations = 0
            file_conversions = 0
            file_drops = 0

            for line_num, line in enumerate(lines, 1):
                raw = line.rstrip('\n')
                line = raw.strip()
                if not line:
                    # 保持空行
                    continue

                try:
                    parts = line.split()
                    if len(parts) >= 1:
                        old_class_id = int(parts[0])
                        file_annotations += 1

                        if old_class_id in self.class_remapping:
                            # 转换并保留
                            new_class_id = self.class_remapping[old_class_id]
                            parts[0] = str(new_class_id)
                            new_line = ' '.join(parts) + '\n'
                            converted_lines.append(new_line)
                            file_conversions += 1
                        else:
                            # 不在映射中的类别 -> 删除（不写入）
                            file_drops += 1
                    else:
                        # 格式错误，视为无效并跳过
                        continue
                except ValueError as e:
                    print(f"    警告: 文件 {filename} 第{line_num}行解析失败: {raw} - {e}")
                    continue

            # 写回文件（只写已转换且保留的标注）
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(converted_lines)

            # 更新统计
            self.stats['total_annotations'] += file_annotations
            self.stats['converted_annotations'] += file_conversions
            self.stats['dropped_annotations'] += file_drops

            # 文件级别输出
            parts_msgs = []
            if file_conversions > 0:
                parts_msgs.append(f"{file_conversions} 个标注已转换")
            if file_drops > 0:
                parts_msgs.append(f"{file_drops} 个标注被删除")

            if parts_msgs:
                print(f"  --- {' / '.join(parts_msgs)} （共 {file_annotations} 个标注）")
            else:
                print(f"  - 无需要转换或删除的标注")

            return True

        except Exception as e:
            print(f"  ✗ 转换文件 {filename} 失败: {e}")
            return False

    def run_conversion(self):
        """运行批量转换"""
        print("开始批量转换...")
        print("-" * 60)

        # 创建备份目录
        if not self.create_backup():
            print("警告: 备份目录创建失败，继续执行...")

        # 获取所有txt文件，跳过之前记录的 classification 文件（按绝对路径比较）
        all_txt = glob.glob(os.path.join(self.labels_dir, "*.txt"))
        txt_files = []
        for p in all_txt:
            try:
                if hasattr(self, 'classification_path') and os.path.abspath(p) == os.path.abspath(self.classification_path):
                    # 跳过 classification 文件
                    print(f"  跳过分类文件: {os.path.basename(p)}")
                    continue
            except Exception:
                pass
            txt_files.append(p)
        self.stats['total_files'] = len(txt_files)

        if self.stats['total_files'] == 0:
            print("未找到任何.txt标签文件")
            return

        print(f"找到 {self.stats['total_files']} 个标签文件")
        print("开始转换...")
        print("-" * 60)

        # 转换每个文件
        for i, txt_file in enumerate(txt_files, 1):
            print(f"[{i}/{self.stats['total_files']}] 处理: {os.path.basename(txt_file)}", end='')
            if self.convert_single_file(txt_file):
                self.stats['processed_files'] += 1
            else:
                self.stats['skipped_files'] += 1

        # 显示统计结果
        self.show_statistics()

    def show_statistics(self):
        """显示转换统计结果"""
        print("=" * 80)
        print("转换完成！")
        print("=" * 80)

        print(f"统计结果:")
        print(f"  总文件数: {self.stats['total_files']}")
        print(f"  成功处理: {self.stats['processed_files']}")
        print(f"  跳过文件: {self.stats['skipped_files']}")
        print(f"  总标注数: {self.stats['total_annotations']:,}")
        print(f"  转换标注: {self.stats['converted_annotations']:,}")
        print(f"  删除标注: {self.stats['dropped_annotations']:,}")

        if self.stats['total_annotations'] > 0:
            conversion_rate = self.stats['converted_annotations'] / self.stats['total_annotations'] * 100
            print(f"  转换比例: {conversion_rate:.2f}%")

        if self.backup_dir:
            print(f"\n备份目录: {self.backup_dir}")
            print("如需恢复，请将备份文件复制回原目录")


if __name__ == "__main__":
    main()


