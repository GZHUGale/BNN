import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser()
# 数据集的根目录
parser.add_argument('--data-path', default='C:/Users/MECHREVO1/Desktop/opt_sar_label/', help='dataset')

args = parser.parse_args()
print(args)

tree_root = args.data_path
opt_img_root_path = os.path.join(tree_root, 'opt')
sar_img_root_path = os.path.join(tree_root, 'sar')

filename = os.listdir(opt_img_root_path)
Z = np.zeros((256, 256))
for n in filename:
    opt_img_path = os.path.join(opt_img_root_path, n)
    sar_img_path = os.path.join(sar_img_root_path, n)
    # opt转灰度图
    opt_img_path_ = Image.open(opt_img_path).convert('L')
    sar_img_path_ = Image.open(sar_img_path)

    opt_img = np.array(opt_img_path_, dtype='uint8')
    sar_img = np.array(sar_img_path_, dtype='uint8')
    # print('{}图片'.format(n.split('.')[0]))
    if sar_img.ndim == 3:
        print(filename)
    Size = opt_img.size  # 图片尺寸
    w = opt_img_path_.width  # 图片宽
    h = opt_img_path_.height  # 图片高

    # 生成全0统计矩阵
    # Z = np.zeros((256, 256))
    # print(n)
    for i in range(w):
        for j in range(h):
            count = 0
            opt_index = [i, j]
            sar_index = opt_index
            opt_num = opt_img[opt_index[0], opt_index[1]]
            sar_num = sar_img[sar_index[0], sar_index[1]]

            # 生成统计矩阵
            Z[sar_num, opt_num] = np.add(Z[sar_num, opt_num], 1)

np.savetxt("C:/Users/MECHREVO1/Desktop/opt_sar_label/total_statistics.txt", Z, fmt="%d")
# a = pd.DataFrame(Z)
# print(a)
