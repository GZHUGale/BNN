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
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

parser = argparse.ArgumentParser()
# 训练数据集的根目录
parser.add_argument('--data-path', default='C:/Users/MECHREVO1/Desktop/opt_sar_label/', help='dataset')
parser.add_argument('--only-ship', default=True, help='默认只训练船舶数据，False为训练船舶和非船舶数据')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
args = parser.parse_args()
print(args)
root_path = "C:/Users/MECHREVO1/Desktop/opt_sar_label/"
save_result_path = "C:/Users/MECHREVO1/Desktop/fg"


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        if args.only_ship:
            self.class_names = ['ship']
        else:
            self.class_names = ['ship', 'noship']
        self.num_classes = len(self.class_names)
        self.sar_data, self.opt_data, self.labels, self.img_id = self.load_data()

    def load_data(self):
        sar_imgs = []
        opt_imgs = []

        labels = []
        img_id = []

        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, class_name)
            sar_root = os.path.join(class_path, 'sar')
            opt_root = os.path.join(class_path, 'opt')

            for i in os.listdir(sar_root):
                opt_img_path = os.path.join(opt_root, i)
                sar_img_path = os.path.join(sar_root, i)

                sar_img = Image.open(sar_img_path)
                opt_img = Image.open(opt_img_path).convert('L')

                sar_array = np.array(sar_img, dtype='uint8')
                opt_array = np.array(opt_img, dtype='uint8')
                sar_imgs.append(sar_array)
                opt_imgs.append(opt_array)

                labels.append(class_idx)
                img_id.append(np.array(int(i.split('.')[0])))

        # Convert labels to one-hot encoding
        labels = np.array(labels)
        labels = torch.tensor(labels).long()
        labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()

        return sar_imgs, opt_imgs, labels, img_id

    def __len__(self):
        return len(self.sar_data)

    def __getitem__(self, idx):
        sar_image = self.sar_data[idx]
        opt_image = self.opt_data[idx]
        label = self.labels[idx]
        image_id = self.img_id[idx]
        if self.transform is not None:
            sar_image = self.transform(sar_image)
            opt_image = self.transform(opt_image)

        return sar_image, opt_image, label, image_id


class BridgeNet(nn.Module):
    # def __init__(self, sar_p, sar_n, sar_z, opt_p, opt_n, opt_z):
    def __init__(self):
        super(BridgeNet, self).__init__()
        self.res1 = models.resnet50(weights=None)
        self.res2 = models.resnet50(weights=None)
        self.res1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.res2.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.res1_fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.Tanh(),
        )
        self.res2_fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.Tanh(),
        )

    def forward(self, x1, x2, image_id):  # x1 -> sar, x2 -> opt
        # sar_seg, opt_seg = self.segment(x1, x2)
        sar = self.res1_fc(self.res1(x1))
        opt = self.res2_fc(self.res2(x2))
        return sar, opt

    # def segment(self, image1, image2):
    #     # 切片参数
    #     patch_size = (32, 32)
    #     stride = (16, 16)
    #     # 使用torch.nn.functional.unfold进行切片
    #     patches1 = F.unfold(image1, patch_size, stride=stride)
    #     patches2 = F.unfold(image2, patch_size, stride=stride)
    #     return patches1, patches2


def neg_hscore(f, g):
    f0 = f - torch.mean(f, 0)
    g0 = g - torch.mean(g, 0)
    corr = torch.mean(torch.sum(f0 * g0, 1))
    cov_f = torch.mm(torch.t(f0), f0) / (f0.size()[0] - 1.)
    cov_g = torch.mm(torch.t(g0), g0) / (g0.size()[0] - 1.)
    result = - corr + torch.trace(torch.mm(cov_f, cov_g)) / 2.
    if result > 1000:
        # 保存
        pass
    return result


def normalize(object):
    object_max = object.max()
    object_min = object.min()
    normal = np.divide(np.subtract(object, object_min), np.subtract(object_max, object_min))
    return normal

def hotmap(fx, gy, image_id):
    fx = fx - torch.mean(fx, 0)
    gy = gy - torch.mean(gy, 0)
    fx = fx.cpu().detach().numpy()
    gy = gy.cpu().detach().numpy()
    image_id = image_id.cpu().detach().numpy()
    for id in range(len(image_id)):
        id_ = str(image_id[id])
        sar_image_path = os.path.join(root_path, 'ship', 'sar', id_ + '.png')
        opt_image_path = os.path.join(root_path, 'ship', 'opt', id_ + '.png')
        sar_image = Image.open(sar_image_path)
        opt_image = Image.open(opt_image_path).convert('L')
        sar_image = np.array(sar_image, dtype='uint8')
        opt_image = np.array(opt_image, dtype='uint8')
        w = sar_image.shape[0]  # 图片宽
        h = sar_image.shape[1]  # 图片高
        Z = np.zeros((w, h))
        for m in range(w):
            for n in range(h):
                opt_index = [m, n]
                sar_index = opt_index
                opt_num = opt_image[opt_index[0], opt_index[1]]
                sar_num = sar_image[sar_index[0], sar_index[1]]
                fx_index_data = fx[id][sar_num]
                gy_index_data = gy[id][opt_num]
                fxgy_ = np.dot(fx_index_data, gy_index_data)
                # 热图矩阵
                Z[m, n] = fxgy_

        # np.savetxt('C:/Users/MECHREVO1/Desktop/opt_sar_label/ship/heatmap_txt/{}.txt'.format(filename.split('.')[0]), Z, fmt="%f")
        plt.figure()
        plt.axis('off')
        plt.imshow(Z, cmap='jet')
        plt.savefig('C:/Users/MECHREVO1/Desktop/heatmap/{}.png'.format(int(id_)), bbox_inches='tight', pad_inches=0)
        plt.close()
        # image = normalize(Z) * 255.
        # image = Image.fromarray(image.astype('uint8'))
        # # 保存图片
        # image.save('C:/Users/MECHREVO1/Desktop/heatmap/{}.png'.format(int(id_)))

    return 0


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = ToTensor()
    custom_dataset = CustomDataset(data_path=root_path, transform=transform)
    # custom_dataset = BNN_Dataset(data_root)
    input_tensor = DataLoader(custom_dataset, batch_size=4, shuffle=False)
    model = BridgeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    epochs = args.epochs
    # Enable gradient anomaly detection
    torch.autograd.set_detect_anomaly(True)
    train_step = len(input_tensor)
    all_train_hscore_loss = []
    all_f = []
    all_g = []
    for i in range(epochs):
        model.train()
        train_hscore_loss = 0.0
        for j, (sar_, opt_, labels, image_id_) in enumerate(input_tensor):
            sar, opt, label, image_id = sar_.to(device), opt_.to(device), labels.to(device), image_id_.to(device)
            optimizer.zero_grad()
            f, g = model(sar, opt, image_id)
            # std_f = torch.std(f, dim=1).unsqueeze(1)
            # std_g = torch.std(g, dim=1).unsqueeze(1)
            # f = f / std_f
            # g = g / std_g
            # 循环迭代到最后一次才保存图片
            if i == epochs - 1:
                a = hotmap(f, g, image_id)
            hscore_loss = neg_hscore(f, g)
            hscore_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_hscore_loss += hscore_loss.item()
        print(train_hscore_loss/train_step)
            # all_g.append(g)
            # all_f.append(f)
            # final_all = ['f', all_f, 'g', all_g]
            # with open(save_result_path, 'wb') as fp:
            #     pickle.dump(final_all, fp)

