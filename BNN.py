import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision import models
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
parser = argparse.ArgumentParser()
# 训练数据集的根目录
parser.add_argument('--data-path', default='C:/Users/MECHREVO1/Desktop/opt_sar_label/', help='dataset')
parser.add_argument('--only-ship', default=True, help='默认只训练船舶数据，False为训练船舶和非船舶数据')
parser.add_argument('--save-convpicture', default=False, help='默认保存第一层卷积后的特征图包含正负零特征')
parser.add_argument('--save-model', default=False, help='默认不保存训练权重')
parser.add_argument('--heatmap-kernel', default=7)
parser.add_argument('--save-pnz', default=False, help='默认保存三通道图片，False保存4通道图片')
# 训练的总epoch数
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
args = parser.parse_args()
print(args)


# 数据加载
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        if args.only_ship:
            self.class_names = ['ship']
        else:
            self.class_names = ['ship', 'noship']
        self.num_classes = len(self.class_names)
        self.sar_data,self.opt_data,self.heatmap_data, self.labels, self.img_id = self.load_data()

    def load_data(self):
        sar_imgs = []
        opt_imgs = []
        heatmap_imgs = []
        labels = []
        img_id = []

        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, class_name)
            sar_root = os.path.join(class_path, 'sar')
            opt_root = os.path.join(class_path, 'opt')
            heatmap_root = os.path.join(class_path, 'heatmap_txt')
            for i in os.listdir(sar_root):
                opt_img_path = os.path.join(opt_root, i)
                sar_img_path = os.path.join(sar_root, i)
                heatmap_txt_path = os.path.join(heatmap_root, i.split('.')[0] + ".txt")
                sar_img = Image.open(sar_img_path)
                opt_img = Image.open(opt_img_path).convert('L')
                heatmap_txt = np.loadtxt(heatmap_txt_path)
                sar_array = np.array(sar_img, dtype='uint8')
                opt_array = np.array(opt_img, dtype='uint8')
                sar_imgs.append(sar_array)
                opt_imgs.append(opt_array)
                heatmap_imgs.append(heatmap_txt)
                labels.append(class_idx)
                img_id.append(np.array(int(i.split('.')[0])))

        # Convert labels to one-hot encoding
        labels = np.array(labels)
        labels = torch.tensor(labels).long()
        labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()

        return sar_imgs, opt_imgs, heatmap_imgs, labels, img_id

    def __len__(self):
        return len(self.sar_data)

    def __getitem__(self, idx):
        sar_image = self.sar_data[idx]
        opt_image = self.opt_data[idx]
        hot_image = self.heatmap_data[idx]
        label = self.labels[idx]
        image_id = self.img_id[idx]
        if self.transform is not None:
            sar_image = self.transform(sar_image)
            opt_image = self.transform(opt_image)
            hot_image = self.transform(hot_image)

        return sar_image, opt_image, hot_image, label, image_id

def normalize(object):
    object_max = object.max()
    object_min = object.min()
    normal = np.divide(np.subtract(object, object_min), np.subtract(object_max, object_min))
    return normal


def normalize_torch(object):
    obj_max = object.max()
    obj_min = object.min()
    normal = torch.divide(torch.sub(object, obj_min), torch.sub(obj_max, obj_min))
    return normal

# def test_test_picture(h_f1_p, h_f1_n, h_f1_z, h_f2_p_index, h_f2_n_index, h_f2_z_index, id):
#     # FE1
#     h_f1_p_array = h_f1_p[0, 0, :, :].cpu().detach().numpy()
#     h_f1_n_array = h_f1_n[0, 0, :, :].cpu().detach().numpy()
#     h_f1_z_array = h_f1_z[0, 0, :, :].cpu().detach().numpy()
#     # h_f1_p_array[h_f1_p_array != 0] = 1
#     # h_f1_n_array[h_f1_n_array != 0] = 1
#     # h_f1_z_array[h_f1_z_array != 0] = 1
#     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f1_p/{}_h_f1_p.txt'.format(id.item()), h_f1_p_array, fmt="%f")
#     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f1_n/{}_h_f1_n.txt'.format(id.item()), h_f1_n_array, fmt="%f")
#     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f1_z/{}_h_f1_z.txt'.format(id.item()), h_f1_z_array, fmt="%f")
#     # FE2_index
#     h_f2_p_index_array = h_f2_p_index[0, 0, :, :].cpu().detach().numpy()
#     h_f2_n_index_array = h_f2_n_index[0, 0, :, :].cpu().detach().numpy()
#     h_f2_z_index_array = h_f2_z_index[0, 0, :, :].cpu().detach().numpy()
#     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f2_p_index/{}_h_f2_p_index.txt'.format(id.item()), h_f2_p_index_array, fmt="%f")
#     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f2_n_index/{}_h_f2_n_index.txt'.format(id.item()), h_f2_n_index_array, fmt="%f")
#     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f2_z_index/{}_h_f2_z_index.txt'.format(id.item()), h_f2_z_index_array, fmt="%f")
#     return 0

# 定义三桥式网络
class three_BridgeNet(nn.Module):
    # def __init__(self, sar_p, sar_n, sar_z, opt_p, opt_n, opt_z):
    def __init__(self, p_net, n_net, z_net):
        super(three_BridgeNet, self).__init__()
        self.heatmap_kernel_size = args.heatmap_kernel
        # self.sar_resnet_p = sar_p
        # self.sar_resnet_n = sar_n
        # self.sar_resnet_z = sar_z
        # self.opt_resnet_p = opt_p
        # self.opt_resnet_n = opt_n
        # self.opt_resnet_z = opt_z

        self.p_net = p_net
        self.n_net = n_net
        self.z_net = z_net

        # 第一层卷积
        self.heatmap_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.heatmap_kernel_size, stride=1, padding=0)
        )

        self.sar_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=self.heatmap_kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(1,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        )

        self.opt_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=self.heatmap_kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(1,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        )

        self.p_fc = nn.Sequential(
            nn.Linear(1000, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.n_fc = nn.Sequential(
            nn.Linear(1000, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.z_fc = nn.Sequential(
            nn.Linear(1000, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, image_id, train=True):  # x1 -> sar , x2 -> opt, x3 -> heatmap
        x1_ = self.sar_conv(x1)
        x2_ = self.opt_conv(x2)
        x3_ = self.heatmap_pool(x3)
        x3_shape = x3.shape
        if args.save_convpicture:
            if train:
                sar_p, sar_n, sar_z, opt_p, opt_n, opt_z = self.pnz(x1_, x2_, x3_, image_id, save_conv_picture=False)
            else:
                sar_p, sar_n, sar_z, opt_p, opt_n, opt_z = self.pnz(x1_, x2_, x3_, image_id, save_conv_picture=True)
        else:
            if train:
                sar_p, sar_n, sar_z, opt_p, opt_n, opt_z = self.mask_bbox(x1_, x2_, x3_shape, x3_, image_id,
                                                                          save_original_picture=True)
            else:
                sar_p, sar_n, sar_z, opt_p, opt_n, opt_z = self.mask_bbox(x1_, x2_, x3_shape, x3_, image_id,
                                                                          save_original_picture=True)
        # else:
        #     sar_p, sar_n, sar_z, opt_p, opt_n, opt_z = self.mask_bbox(x1, x2, x1_, x2_, x3_, image_id, save_picture=True)
        # sar_p, sar_n, sar_z, opt_p, opt_n, opt_z = self.pnz(x1_, x2_, x3_)
        # if train:
        #     sar_p, sar_n, sar_z, opt_p, opt_n, opt_z = self.Fixed_threshold_partition(
        #         x1_, x2_, x3_, image_id, save_picture=False)
        # else:
        #     sar_p, sar_n, sar_z, opt_p, opt_n, opt_z = self.Fixed_threshold_partition(
        #         x1_, x2_, x3_, image_id, save_picture=True)
        # un_sp, un_op = self.unpool_test(sar_p, opt_p)
        # h = self.Class_activation_diagram(h_m, x2)
        p = self.p_fc(self.p_net(torch.cat((sar_p, opt_p), dim=1)))
        n = self.n_fc(self.n_net(torch.cat((sar_n, opt_n), dim=1)))
        z = self.z_fc(self.z_net(torch.cat((sar_z, opt_z), dim=1)))
        return p, n, z

    def pnz(self, sar_conv, opt_conv, heatmap_avg, image_id, save_conv_picture=False):
        # matplotlib.use('Qt5Agg')
        sar_conv_normal = normalize_torch(sar_conv)
        opt_conv_normal = normalize_torch(opt_conv)
        # heatmap_f1_p = normalize_torch(heatmap_avg)
        # heatmap_f1_n = normalize_torch(torch.mul(heatmap_avg, -1))
        # heatmap_f1_z = normalize_torch(torch.sub(1, torch.abs(heatmap_avg)))
        # threshold = 0.5
        heatmap_f1_p = heatmap_avg
        heatmap_f1_n = torch.mul(heatmap_avg, -1)
        h_max = heatmap_f1_p.max()
        h_min = heatmap_f1_p.min()
        pcoef = 0.66
        bdet = ((h_max - h_min) / 20) * 0.5
        threshold_bias = (h_max + h_min) / 2
        threshold_p = (h_max - h_min) * pcoef + h_min + bdet  # 0.7 效果比较好
        threshold_n = threshold_p - 2 * threshold_bias
        threshold_z = 1 - threshold_p + threshold_bias + 2 * bdet
        heatmap_f1_z = torch.sub(1, torch.abs(torch.sub(heatmap_avg, threshold_bias)))  # 1-|x-th_bias|

        # heatmap_f1_p[heatmap_f1_p > threshold * 1.2] = 1
        # heatmap_f1_p[heatmap_f1_p <= threshold * 1.2] = 0
        #
        # heatmap_f1_n[heatmap_f1_n > threshold * 1.2] = 1
        # heatmap_f1_n[heatmap_f1_n <= threshold * 1.2] = 0
        #
        # heatmap_f1_z[heatmap_f1_z > threshold * 1.2] = 1
        # heatmap_f1_z[heatmap_f1_z <= threshold * 1.2] = 0

        heatmap_f1_p[heatmap_f1_p < threshold_p] = 0
        heatmap_f1_p[heatmap_f1_p != 0] = 1

        heatmap_f1_n[heatmap_f1_n < threshold_n] = 0
        heatmap_f1_n[heatmap_f1_n != 0] = 1

        heatmap_f1_z[heatmap_f1_z < threshold_z] = 0
        heatmap_f1_z[heatmap_f1_z != 0] = 1


        sar_p = torch.as_tensor(torch.mul(sar_conv_normal, heatmap_f1_p), dtype=torch.float).clone()
        sar_n = torch.as_tensor(torch.mul(sar_conv_normal, heatmap_f1_n), dtype=torch.float).clone()
        sar_z = torch.as_tensor(torch.mul(sar_conv_normal, heatmap_f1_z), dtype=torch.float).clone()
        opt_p = torch.as_tensor(torch.mul(opt_conv_normal, heatmap_f1_p), dtype=torch.float).clone()
        opt_n = torch.as_tensor(torch.mul(opt_conv_normal, heatmap_f1_n), dtype=torch.float).clone()
        opt_z = torch.as_tensor(torch.mul(opt_conv_normal, heatmap_f1_z), dtype=torch.float).clone()
        if save_conv_picture:
            for b in range(heatmap_avg.shape[0]):
                for c in range(heatmap_avg.shape[1]):
                    id = image_id[b].cpu().detach().numpy()
                    heatmap_positive = heatmap_f1_p[b, c, :, :].squeeze().cpu().detach().numpy() * 255.
                    heatmap_negative = heatmap_f1_n[b, c, :, :].squeeze().cpu().detach().numpy() * 255.
                    heatmap_zero = heatmap_f1_z[b, c, :, :].squeeze().cpu().detach().numpy() * 255.
                    sar_positive = sar_p[b, c, :, :].squeeze().cpu().detach().numpy() * 255.
                    sar_negative = sar_n[b, c, :, :].squeeze().cpu().detach().numpy() * 255.
                    sar_zero = sar_z[b, c, :, :].squeeze().cpu().detach().numpy() * 255.
                    opt_positive = opt_p[b, c, :, :].squeeze().cpu().detach().numpy() * 255.
                    opt_negative = opt_n[b, c, :, :].squeeze().cpu().detach().numpy() * 255.
                    opt_zero = opt_z[b, c, :, :].squeeze().cpu().detach().numpy() * 255.

                    sar_p_image = Image.fromarray(sar_positive.astype('uint8'))
                    sar_n_image = Image.fromarray(sar_negative.astype('uint8'))
                    sar_z_image = Image.fromarray(sar_zero.astype('uint8'))
                    opt_p_image = Image.fromarray(opt_positive.astype('uint8'))
                    opt_n_image = Image.fromarray(opt_negative.astype('uint8'))
                    opt_z_image = Image.fromarray(opt_zero.astype('uint8'))
                    heatmap_p_image = Image.fromarray(heatmap_positive.astype('uint8'))
                    heatmap_n_image = Image.fromarray(heatmap_negative.astype('uint8'))
                    heatmap_z_image = Image.fromarray(heatmap_zero.astype('uint8'))

                    sar_p_image.save('C:/Users/MECHREVO1/Desktop/conv_picture/sar_positive/{}.png'.format(id.item()))
                    sar_n_image.save('C:/Users/MECHREVO1/Desktop/conv_picture/sar_negative/{}.png'.format(id.item()))
                    sar_z_image.save('C:/Users/MECHREVO1/Desktop/conv_picture/sar_zero/{}.png'.format(id.item()))
                    opt_p_image.save('C:/Users/MECHREVO1/Desktop/conv_picture/opt_positive/{}.png'.format(id.item()))
                    opt_n_image.save('C:/Users/MECHREVO1/Desktop/conv_picture/opt_negative/{}.png'.format(id.item()))
                    opt_z_image.save('C:/Users/MECHREVO1/Desktop/conv_picture/opt_zero/{}.png'.format(id.item()))
                    heatmap_p_image.save('C:/Users/MECHREVO1/Desktop/conv_picture/heatmap_positive/{}.png'.format(id.item()))
                    heatmap_n_image.save('C:/Users/MECHREVO1/Desktop/conv_picture/heatmap_negative/{}.png'.format(id.item()))
                    heatmap_z_image.save('C:/Users/MECHREVO1/Desktop/conv_picture/heatmap_zero/{}.png'.format(id.item()))
        return sar_p, sar_n, sar_z, opt_p, opt_n, opt_z

    def mask_bbox(self, sar_conv, opt_conv, heatmap_shape, heatmap_avg, img_index, save_original_picture=True):
        # matplotlib.use('Qt5Agg')
        # heatmap_f1_p = normalize_torch(heatmap_avg)
        # heatmap_f1_n = normalize_torch(torch.mul(heatmap_avg, -1))
        # heatmap_f1_z = normalize_torch(torch.sub(1, torch.abs(heatmap_avg)))
        heatmap_f1_p = heatmap_avg
        heatmap_f1_n = torch.mul(heatmap_avg, -1)
        h_max = heatmap_f1_p.max()
        h_min = heatmap_f1_p.min()
        pcoef = 0.66
        bcoef = 0.05
        bdet = (h_max - h_min) * bcoef
        threshold_bias = (h_max + h_min) / 2
        threshold_p = (h_max - h_min) * pcoef + h_min + bdet  # 0.7 效果比较好
        threshold_n = threshold_p - 2 * threshold_bias
        threshold_z = 1 - threshold_p + threshold_bias + 2 * bdet
        heatmap_f1_z = torch.sub(1, torch.abs(torch.sub(heatmap_avg, threshold_bias)))  # 1-|x-th_bias|
        # heatmap_f1_z = torch.abs(torch.sub(heatmap_avg, threshold_bias))
        # heatmap_f1_z = torch.sub(h_max, torch.abs(torch.sub(heatmap_avg, threshold_bias)))  # max-|x-th_bias|
        maxpool = nn.MaxPool2d(self.heatmap_kernel_size, stride=1, return_indices=True)
        heatmap_f2_maxpool_p, h_p_index = maxpool(heatmap_f1_p)
        heatmap_f2_maxpool_n, h_n_index = maxpool(heatmap_f1_n)
        heatmap_f2_maxpool_z, h_z_index = maxpool(heatmap_f1_z)
        # batch_size = heatmap_avg.shape[0]
        # num_channels = heatmap_avg.shape[1]
        # fixation = heatmap_avg.shape[3]
        # # 将扁平化的索引值映射回 [0, 249] 范围
        # h_p_col_index = h_p_index // fixation
        # h_p_row_index = h_p_index % fixation
        # h_n_col_index = h_n_index // fixation
        # h_n_row_index = h_n_index % fixation
        # h_z_col_index = h_z_index // fixation
        # h_z_row_index = h_z_index % fixation
        # batch_indices = torch.arange(batch_size).view(batch_size, 1, 1, 1).expand(batch_size, 1, 244, 244)
        # channel_indices = torch.arange(num_channels).view(1, num_channels, 1, 1).expand(1, num_channels, 244, 244)
        # h_p_row_indices = h_p_row_index
        # h_n_row_indices = h_n_row_index
        # h_z_row_indices = h_z_row_index
        # h_p_col_indices = h_p_index
        # h_n_col_indices = h_n_index
        # h_z_col_indices = h_z_index
        # sar_p_extracted_images = sar_conv[batch_indices, channel_indices, h_p_row_index, h_p_col_index]
        # sar_n_extracted_images = sar_conv[batch_indices, channel_indices, h_n_row_index, h_n_col_index]
        # sar_z_extracted_images = sar_conv[batch_indices, channel_indices, h_z_row_index, h_z_col_index]
        # opt_p_extracted_images = opt_conv[batch_indices, channel_indices, h_p_row_index, h_p_col_index]
        # opt_n_extracted_images = opt_conv[batch_indices, channel_indices, h_n_row_index, h_n_col_index]
        # opt_z_extracted_images = opt_conv[batch_indices, channel_indices, h_z_row_index, h_z_col_index]
        # if save_original_picture:
        #     if args.save_pnz:
        #         sar_p_normal = normalize_torch(sar_p_extracted_images) * 255.
        #         sar_n_normal = normalize_torch(sar_n_extracted_images) * 255.
        #         sar_z_normal = normalize_torch(sar_z_extracted_images) * 255.
        #         opt_p_normal = normalize_torch(opt_p_extracted_images) * 255.
        #         opt_n_normal = normalize_torch(opt_n_extracted_images) * 255.
        #         opt_z_normal = normalize_torch(opt_z_extracted_images) * 255.
        #         sar_three = torch.cat((sar_p_normal, sar_n_normal, sar_z_normal), dim=1)
        #         opt_three = torch.cat((opt_p_normal, opt_n_normal, opt_z_normal), dim=1)
        #         sar_three_array = np.transpose(sar_three.squeeze().cpu().detach().numpy(), (2, 1, 0))
        #         opt_three_array = np.transpose(opt_three.squeeze().cpu().detach().numpy(), (2, 1, 0))
        #         sar_three_image = Image.fromarray(sar_three_array.astype('uint8'), mode='RGB')
        #         opt_three_image = Image.fromarray(opt_three_array.astype('uint8'), mode='RGB')
        #         sar_three_image.save('C:/Users/MECHREVO1/Desktop/three_picture/sar/{}.png'.format(img_index.item()))
        #         opt_three_image.save('C:/Users/MECHREVO1/Desktop/three_picture/opt/{}.png'.format(img_index.item()))
        #     else:
        #         sar_p_normal = normalize_torch(sar_p_extracted_images) * 255.
        #         sar_n_normal = normalize_torch(sar_n_extracted_images) * 255.
        #         opt_p_normal = normalize_torch(opt_p_extracted_images) * 255.
        #         opt_n_normal = normalize_torch(opt_n_extracted_images) * 255.
        #         four_picture = torch.cat((sar_p_normal, opt_p_normal, sar_n_normal, opt_n_normal), dim=1)
        #         four_picture_array = np.transpose(four_picture.squeeze().cpu().detach().numpy(), (2, 1, 0))
        #         four_picture_image = Image.fromarray(four_picture_array.astype('uint8'), mode='RGBA')
        #         four_picture_image.save('C:/Users/MECHREVO1/Desktop/four_picture/{}.png'.format(img_index.item()))
        #
        # return sar_p_extracted_images, sar_n_extracted_images, sar_z_extracted_images, \
        #     opt_p_extracted_images, opt_n_extracted_images, opt_z_extracted_images


        mask_positive_zeros = torch.zeros(heatmap_shape).to(device)
        mask_negative_zeros = torch.zeros(heatmap_shape).to(device)
        mask_zero_zeros = torch.zeros(heatmap_shape).to(device)
        # sar_positive = torch.zeros(sar.shape).to(device)
        # sar_negative = torch.zeros(sar.shape).to(device)
        # sar_zero = torch.zeros(sar.shape).to(device)
        # opt_positive = torch.zeros(opt.shape).to(device)
        # opt_negative = torch.zeros(opt.shape).to(device)
        # opt_zero = torch.zeros(opt.shape).to(device)
        # if save_original_picture:
        #     h_f1_p_array = heatmap_f1_p[0, 0, :, :].cpu().detach().numpy()
        #     h_f1_n_array = heatmap_f1_n[0, 0, :, :].cpu().detach().numpy()
        #     h_f1_z_array = heatmap_f1_z[0, 0, :, :].cpu().detach().numpy()
        #     # h_f1_p_array[h_f1_p_array != 0] = 1
        #     # h_f1_n_array[h_f1_n_array != 0] = 1
        #     # h_f1_z_array[h_f1_z_array != 0] = 1
        #     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f1_p/{}_h_f1_p.txt'.format(img_index.item()), h_f1_p_array, fmt="%f")
        #     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f1_n/{}_h_f1_n.txt'.format(img_index.item()), h_f1_n_array, fmt="%f")
        #     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f1_z/{}_h_f1_z.txt'.format(img_index.item()), h_f1_z_array, fmt="%f")
        #     # FE2_index
        #     h_f2_p_index_array = h_p_index[0, 0, :, :].cpu().detach().numpy()
        #     h_f2_n_index_array = h_n_index[0, 0, :, :].cpu().detach().numpy()
        #     h_f2_z_index_array = h_z_index[0, 0, :, :].cpu().detach().numpy()
        #     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f2_p_index/{}_h_f2_p_index.txt'.format(img_index.item()), h_f2_p_index_array, fmt="%f")
        #     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f2_n_index/{}_h_f2_n_index.txt'.format(img_index.item()), h_f2_n_index_array, fmt="%f")
        #     np.savetxt('C:/Users/MECHREVO1/Desktop/picture_adnormal/h_f2_z_index/{}_h_f2_z_index.txt'.format(img_index.item()), h_f2_z_index_array, fmt="%f")

        all_one = torch.ones(self.heatmap_kernel_size, self.heatmap_kernel_size).to(device)
        fixation = heatmap_avg.shape[3]
        for b in range(h_p_index.shape[0]):
            for c in range(h_p_index.shape[1]):
                # positive_zero = torch.zeros(256, 256).to(device)
                # negative_zero = torch.zeros(256, 256).to(device)
                # zero_zero = torch.zeros(256, 256).to(device)
                for w in range(h_p_index.shape[2]):
                    for h in range(h_p_index.shape[3]):
                        # 正样本
                        a_p = h_p_index[b, c, w, h]
                        m_p = int(a_p / fixation)
                        n_p = a_p % fixation
                        h_value_p = heatmap_f1_p[b, c, m_p, n_p]
                        if h_value_p > threshold_p:
                            mask_positive_zeros[b, c, m_p:m_p + self.heatmap_kernel_size,
                            n_p:n_p + self.heatmap_kernel_size] = torch.add(
                                mask_positive_zeros[b, c, m_p:m_p + self.heatmap_kernel_size,
                                n_p:n_p + self.heatmap_kernel_size], all_one)
                        # 负样本
                        a_n = h_n_index[b, c, w, h]
                        m_n = int(a_n / fixation)
                        n_n = a_n % fixation
                        h_value_n = heatmap_f1_n[b, c, m_n, n_n]
                        if h_value_n > threshold_n:
                            mask_negative_zeros[b, c, m_n:m_n + self.heatmap_kernel_size,
                            n_n:n_n + self.heatmap_kernel_size] = torch.add(
                                mask_negative_zeros[b, c, m_n:m_n + self.heatmap_kernel_size,
                                n_n:n_n + self.heatmap_kernel_size], all_one)
                        # 零样本
                        a_z = h_z_index[b, c, w, h]
                        m_z = int(a_z / fixation)
                        n_z = a_z % fixation
                        h_value_z = heatmap_f1_z[b, c, m_z, n_z]
                        if h_value_z > threshold_z:
                            mask_zero_zeros[b, c, m_z:m_z + self.heatmap_kernel_size,
                            n_z:n_z + self.heatmap_kernel_size] = torch.add(
                                mask_zero_zeros[b, c, m_z:m_z + self.heatmap_kernel_size,
                                n_z:n_z + self.heatmap_kernel_size], all_one)

                if save_original_picture:
                    id = img_index[b].cpu().detach().numpy()
                    ppp = mask_positive_zeros[b, c, :, :].cpu().detach().numpy()
                    nnn = mask_negative_zeros[b, c, :, :].cpu().detach().numpy()
                    zzz = mask_zero_zeros[b, c, :, :].cpu().detach().numpy()

                    # # 保存彩色图
                    # plt.figure()
                    # plt.axis('off')
                    # plt.imshow(ppp, cmap='jet')
                    # plt.savefig('C:/Users/MECHREVO1/Desktop/original_picture/mask_positive_color/{}.png'.format(id), bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    # plt.figure()
                    # plt.axis('off')
                    # plt.imshow(nnn, cmap='jet')
                    # plt.savefig('C:/Users/MECHREVO1/Desktop/original_picture/mask_negative_color/{}.png'.format(id), bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    # plt.figure()
                    # plt.axis('off')
                    # plt.imshow(zzz, cmap='jet')
                    # plt.savefig('C:/Users/MECHREVO1/Desktop/original_picture/mask_zero_color/{}.png'.format(id), bbox_inches='tight', pad_inches=0)
                    # plt.close()

                    # 生成正负零蒙版
                    ppp[ppp != 0] = 1
                    nnn[nnn != 0] = 1
                    zzz[zzz != 0] = 1
                    # 保存灰度图
                    p_255 = ppp * 255.
                    n_255 = nnn * 255.
                    z_255 = zzz * 255.
                    mask_p = Image.fromarray(p_255.astype('uint8'))
                    mask_n = Image.fromarray(n_255.astype('uint8'))
                    mask_z = Image.fromarray(z_255.astype('uint8'))
                    mask_p.save('C:/Users/MECHREVO1/Desktop/original_picture/mask_positive/{}.png'.format(id))
                    mask_n.save('C:/Users/MECHREVO1/Desktop/original_picture/mask_negative/{}.png'.format(id))
                    mask_z.save('C:/Users/MECHREVO1/Desktop/original_picture/mask_zero/{}.png'.format(id))

                    sar_array = sar[b, c, :, :].cpu().detach().numpy()
                    opt_array = opt[b, c, :, :].cpu().detach().numpy()
                    sar_p = np.multiply(sar_array, ppp) * 255.
                    sar_n = np.multiply(sar_array, nnn) * 255.
                    sar_z = np.multiply(sar_array, zzz) * 255.
                    opt_p = np.multiply(opt_array, ppp) * 255.
                    opt_n = np.multiply(opt_array, nnn) * 255.
                    opt_z = np.multiply(opt_array, zzz) * 255.

                    # 保存sar和opt分割图
                    sar_p_image = Image.fromarray(sar_p.astype('uint8'))
                    sar_n_image = Image.fromarray(sar_n.astype('uint8'))
                    sar_z_image = Image.fromarray(sar_z.astype('uint8'))
                    opt_p_image = Image.fromarray(opt_p.astype('uint8'))
                    opt_n_image = Image.fromarray(opt_n.astype('uint8'))
                    opt_z_image = Image.fromarray(opt_z.astype('uint8'))

                    sar_p_image.save('C:/Users/MECHREVO1/Desktop/original_picture/sar_positive/{}.png'.format(id))
                    sar_n_image.save('C:/Users/MECHREVO1/Desktop/original_picture/sar_negative/{}.png'.format(id))
                    sar_z_image.save('C:/Users/MECHREVO1/Desktop/original_picture/sar_zero/{}.png'.format(id))

                    opt_p_image.save('C:/Users/MECHREVO1/Desktop/original_picture/opt_positive/{}.png'.format(id))
                    opt_n_image.save('C:/Users/MECHREVO1/Desktop/original_picture/opt_negative/{}.png'.format(id))
                    opt_z_image.save('C:/Users/MECHREVO1/Desktop/original_picture/opt_zero/{}.png'.format(id))

        mask_positive_zeros[mask_positive_zeros != 0] = 1
        mask_negative_zeros[mask_negative_zeros != 0] = 1
        mask_zero_zeros[mask_zero_zeros != 0] = 1

        sar_positive = torch.mul(sar, mask_positive_zeros).clone()
        sar_negative = torch.mul(sar, mask_negative_zeros).clone()
        sar_zero = torch.mul(sar, mask_zero_zeros).clone()
        opt_positive = torch.mul(opt, mask_positive_zeros).clone()
        opt_negative = torch.mul(opt, mask_negative_zeros).clone()
        opt_zero = torch.mul(opt, mask_zero_zeros).clone()

        return sar_positive, sar_negative, sar_zero, opt_positive, opt_negative, opt_zero


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = args.data_path
    # custom_dataset = BNN_Dataset(data_root)
    transform = ToTensor()
    custom_dataset = CustomDataset(data_path=data_root, transform=transform)
    # custom_dataset = BNN_Dataset(data_root)
    input_tensor = DataLoader(custom_dataset, batch_size=1, shuffle=False)
    resnet = models.resnet50(weights=None)
    # 修改第一层卷积层
    resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    p_net = resnet.to(device)
    n_net = resnet.to(device)
    z_net = resnet.to(device)
    # 创建三桥式网络
    model = three_BridgeNet(p_net, n_net, z_net).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    # criterion = BridgingLoss(lambda_positive=1, lambda_negative=1, lambda_zeros=1)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)
    epochs = args.epochs
    # Enable gradient anomaly detection
    torch.autograd.set_detect_anomaly(True)
    train_step = len(input_tensor)
    # save_txt = {}
    for i in range(epochs):
        model.train()
        train_loss = 0.0
        # for j, (sar_, opt_, statistic_, target_) in enumerate(input_tensor):
        #     sars = list(sar.to(device) for sar in sar_)
        #     opts = list(opt.to(device) for opt in opt_)
        #     heatmaps = list(statistic.to(device) for statistic in statistic_)
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in target_]
        #     sar = joint(sars)
        #     opt = joint(opts)
        #     heatmap = joint(heatmaps)
        for j, (sar_, opt_, statistic_, labels, image_id) in enumerate(input_tensor):
            sar, opt, heatmap, label, image_id = sar_.to(
                device), opt_.to(device), statistic_.to(device), labels.to(device), image_id.to(device)
            # sars = list(sar.to(device) for sar in sar_)
            # opts = list(opt.to(device) for opt in opt_)
            # heatmaps = list(statistic.to(device) for statistic in statistic_)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in target_]
            # sar = joint(sars)
            # opt = joint(opts)
            # heatmap = joint(heatmaps)
            # sar, opt, heatmap, target = sar_.to(device), opt_.to(device), statistic_.to(device), target_.to(device)
            optimizer.zero_grad()
            # 输出网络结果
            # p, n, z = model(sar, opt, heatmap)

            p, n, z = model(sar, opt, heatmap, image_id, train=True)
            output = (p + n + z) * 1 / 3
            loss = loss_mse(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if j % 50 == 0:
                print("train epoch[{}/{}] batch_50num:{} loss:{:.3f}".format(i + 1, epochs, (j / 50) + 1,
                                                                             loss.item()))
        print("[epoch %d] train_loss: %.3f" % (i + 1, train_loss / train_step))
    print('save picture successful!')
    # model.eval()
    # with torch.no_grad():
    #     for i, (sar_, opt_, statistic_, labels, image_id) in enumerate(input_tensor):
    #         sar, opt, heatmap, label, image_id = sar_.to(device), opt_.to(device), statistic_.to(device), labels.to(
    #             device), image_id.to(device)
    #         p, n, z = model(sar, opt, heatmap, image_id, train=False)
    #     print('save picture successful!')
    # if args.save_model:
    #     torch.save(model.state_dict(), './backbone/BNN_model_weights_{}.pth'.format(epochs))
