import os

def save_filenames_to_txt(folder_path, output_file):
    # 打开输出文件，准备写入
    with open(output_file, 'w') as f:
        # 遍历文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 将文件名写入txt文件，每个文件名占一行
                f.write(file + '\n')

# 示例用法
folder_path = '/data/lgl/datasets/SYSU-CD/val/GT/'  # 替换为你的文件夹路径
output_file = '/data/lgl/datasets/SYSU-CD/val_list.txt'  # 替换为你想保存的文件名

save_filenames_to_txt(folder_path, output_file)