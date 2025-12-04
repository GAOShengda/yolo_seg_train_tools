import os
import random
import shutil

# 删除文件夹中的图片数量到 750并保留删除的图片
def random_delete_to_750(folder_path):
    # 读取所有图片
    imgs = [f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    imgs.sort()  # 固定顺序，方便连续删

    total = len(imgs)
    print(f"当前图片数量: {total}")

    if total <= 750:
        print("图片数量不足或刚好，无需删除")
        return

    # 需要删除的数量
    need_delete = total - 750
    print(f"需要删除: {need_delete} 张")

    delete_count = 0

    # 用列表标记是否删除
    mark_delete = [False] * total

    while delete_count < need_delete:
        # 随机选择一个起点
        idx = random.randint(0, total - 1)

        # 若该点已被标记删除，跳过
        if mark_delete[idx]:
            continue

        # 随机连续删除 1～8 张
        max_cont = min(8, need_delete - delete_count)   # 不能超过剩余需删
        cont = random.randint(1, max_cont)

        # 对于某些点，连续不足 cont 张，需调整
        end = min(idx + cont, total)

        # 标记删除
        for i in range(idx, end):
            if not mark_delete[i]:
                mark_delete[i] = True
                delete_count += 1
                if delete_count == need_delete:
                    break

    # 创建删除文件夹查看效果
    delete_folder = os.path.join(folder_path, "deleted_preview")
    os.makedirs(delete_folder, exist_ok=True)

    # 删除文件（这里先移动到新文件夹以避免误删）
    for i, flag in enumerate(mark_delete):
        if flag:
            src = os.path.join(folder_path, imgs[i])
            dst = os.path.join(delete_folder, imgs[i])
            shutil.move(src, dst)

    print(f"已成功删除 {delete_count} 张图片，剩余 750 张")
    print(f"被删除的图片已移动到: {delete_folder}")



# ------------------------------
# 程序入口
# ------------------------------
if __name__ == "__main__":
    folder = r"F:\Desktop\residue"     # 修改为你的图片路径
    random_delete_to_750(folder)
