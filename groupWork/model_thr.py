import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image as Img
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
import timm
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import torchvision
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
import pytorch_lightning as pl
import seaborn as sns
import cv2 as cv
import numpy as np
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import math

# 模型名称
MODEL_NAME = 'tf_efficientnet_demo'
# 批次大小
BATCH_SIZE = 64
# 图像大小
IMAGE_SIZE = 900
# 工作线程数
NUM_WORKERS = 15
# 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 是否使用自动混合精度
USE_AMP = True
# 是否初始化
INIT = False

# 输入数据路径
root_in = 'D:\\Demo\input\\dataset'
# 输出数据路径
root_out = 'D:\\Demo\\output'
# 是否有索引
have_index = False

# 类别数
NUM_CLASSES = 100
# 嵌入大小
EMBEDDING_SIZE = 1024
# S, M
S, M = 30.0, 0.5
# 是否容易合并
EASY_MERGING, LS_EPS = False, 0.0


# 定义ArcMarginProduct类
class ArcMarginProduct(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            s: float,
            m: float,
            easy_margin: bool,
            ls_eps: float,
            rank
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.rank = rank

    def forward(self, input: torch.Tensor, label: torch.Tensor, device='cuda') -> torch.Tensor:
        # 计算输入和权重的余弦相似度
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.to(torch.float32)

        # 计算余弦相似度的正弦值
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # 计算余弦相似度与m的乘积
        phi = cosine * self.cos_m - sine * self.sin_m
        # 如果easy_margin为True，则将phi中大于0的部分替换为cosine
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        # 否则，将phi中大于th的部分替换为cosine - mm
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # 创建one-hot编码
        one_hot = torch.zeros(cosine.size(), device=self.rank)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # 如果ls_eps大于0，则对one-hot进行label smoothing
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # 计算输出
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# 定义SorghumModel类
class SorghumModel(pl.LightningModule):
    def __init__(self, model_name, embedding_size, map_location, k_fold, rank, pretrained=True):
        super(SorghumModel, self).__init__()
        self.save_hyperparameters()

        # 创建模型
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=NUM_CLASSES)

        print('load Start!!!')
        # 获取模型分类器的输入特征数
        in_features = self.model.classifier.in_features
        # 将分类器替换为Identity层
        self.model.classifier = nn.Identity()
        # 获取全局池化层
        self.pooling = self.model.global_pool
        # 将全局池化层替换为Identity层
        self.model.global_pool = nn.Identity()
        self.rank = rank
        # 创建多个dropout层
        self.multiple_dropout = [nn.Dropout(0.25) for i in range(8)]
        # 创建线性层
        self.embedding = nn.Linear(in_features * 2, embedding_size)
        # 创建ArcMarginProduct层
        self.fc = ArcMarginProduct(embedding_size,
                                   NUM_CLASSES,
                                   S,
                                   M,
                                   EASY_MERGING,
                                   LS_EPS,
                                   self.rank)

    def forward(self, images, labels):
        # 前向传播
        features = self.model(images)
        # 计算平均池化特征
        pooled_features_avg = self.pooling(features).flatten(1)
        # 计算最大池化特征
        pooled_features_max = nn.AdaptiveMaxPool2d((1, 1))(features).flatten(1)
        # 将平均池化特征和最大池化特征拼接
        pooled_features = torch.cat((pooled_features_avg, pooled_features_max), dim=1)
        # 对拼接后的特征进行多次dropout
        pooled_features_dropout = torch.zeros((pooled_features.shape), device=self.rank)
        for i in range(8):
            pooled_features_dropout += self.multiple_dropout[i](pooled_features)
        # 计算dropout后的平均特征
        pooled_features_dropout /= 8
        # 将平均特征通过线性层得到embedding
        embedding = self.embedding(pooled_features_dropout)
        # 计算输出
        output = self.fc(embedding, labels)
        return output

    def extract(self, images):
        # 前向传播
        features = self.model(images)
        # 计算平均池化特征
        pooled_features_avg = self.pooling(features).flatten(1)
        # 计算最大池化特征
        pooled_features_max = nn.AdaptiveMaxPool2d((1, 1))(features).flatten(1)
        # 将平均池化特征和最大池化特征拼接
        pooled_features = torch.cat((pooled_features_avg, pooled_features_max), dim=1)
        # 将拼接后的特征通过线性层得到embedding
        embedding = self.embedding(pooled_features)
        return embedding

    def training_step(self, batch, batch_idx):
        # 获取batch中的图像和标签
        images, labels = batch
        # 将图像和标签移动到设备上
        images, labels = images.to(self.device), labels.to(self.device)

        # 前向传播
        outputs = self(images, labels)

        # 计算损失
        loss = F.cross_entropy(outputs, labels)

        # 打印训练进度
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 保存模型
        if self.global_step % 100 == 0:
            self.save_model()

        return loss

    def configure_optimizers(self):
        # 创建Adam优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


# 定义Sorghum_Train_Dataset类
class Sorghum_Train_Dataset(Dataset):
    # 初始化函数，传入img_path_csv和df，以及transform
    def __init__(self, img_path_csv='', df=None, transform=None):
        # 如果df不为空，则将df赋值给self.df
        if df is not None:
            self.df = df
        # 否则，从img_path_csv中读取csv文件，并将文件赋值给self.df
        else:
            self.df = pd.read_csv(img_path_csv)
        # 将transform赋值给self.transform
        self.transform = transform

    # 返回数据集的长度
    def __len__(self):
        return self.df.shape[0]

    # 根据索引获取数据集中的数据
    def __getitem__(self, index):
        # 从self.df中获取图片路径
        img = Img.open(os.path.join(root_in, 'train', self.df.iloc[index, 0]))
        # 从self.df中获取标签索引
        label_index = self.df.iloc[index, 4]
        # 如果transform不为空，则对图片进行变换
        if self.transform is not None:
            img = self.transform(img)
        # 返回图片和标签索引
        return img, label_index


# 定义Sorghum_Test_Dataset类
class Sorghum_Test_Dataset(Sorghum_Train_Dataset):
    # 重写getitem方法，用于获取测试集数据
    def __getitem__(self, index):
        # 打开测试集图片
        img = Img.open(os.path.join(root_in, 'test', self.df.iloc[index, 0]))
        # 如果有transform，则对图片进行变换
        if self.transform:
            img = self.transform(img)
        # 返回变换后的图片
        return img


# 定义数据预处理函数
def data_pre_access(file, output):
    # 读取csv文件，将image列作为索引
    labels = pd.read_csv(file, index_col='image')
    # 创建一个空字典，用于存储标签和索引的映射关系
    labels_map = dict()
    # 创建一个全零的numpy数组，用于存储标签索引
    labels['label_index'] = torch.zeros((labels.shape[0])).type(torch.int32).numpy()
    # 遍历labels.cultivar.unique()，将标签和索引的映射关系存储到labels_map中，并将标签索引存储到labels中
    for i, label in enumerate(labels.cultivar.unique()):
        labels_map[i] = label
        labels.loc[labels.cultivar == label, 'label_index'] = i
    # 将labels保存为csv文件
    labels.to_csv(output)

    # 返回标签和索引的映射关系
    return labels_map


if have_index:
    labels_map = {}
    train_df = pd.read_csv(os.path.join(root_out, 'labels_index.csv'), index_col='image')


    def label_f(m):
        labels_map[int(m.label_index)] = m.cultivar


    train_df.apply(label_f, axis=1)
else:
    labels_map = data_pre_access(os.path.join(root_in, 'train_cultivar_mapping.csv'),
                                 output=os.path.join(root_out, 'labels_index.csv'))
    train_df = pd.read_csv(os.path.join(root_out, 'labels_index.csv'), index_col='image')
num_classes = len(labels_map)

num_classes

check_sum = 0
for key, val in tqdm(labels_map.items()):
    train_df[train_df.label_index == key].cultivar.unique() == val
    check_sum += 1

check_sum == len(labels_map)


def predict_test_raw(net, test_iter, device=None):
    # 将网络设置为评估模式
    net.eval()
    # 如果网络是nn.Module类型，则将其设置为评估模式
    if isinstance(net, nn.Module):
        net.eval()
        # 如果没有指定设备，则使用网络参数所在的设备
        if not device:
            device = next(iter(net.parameters())).device
    y = []
    # 将网络移动到指定设备
    net.to(device)
    # 定义softmax函数
    softmax = nn.Softmax(dim=1)
    # 不计算梯度
    with torch.no_grad():
        # 遍历测试数据集
        for X in tqdm(test_iter):
            # 如果X是列表类型，则将列表中的每个元素移动到指定设备
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            # 否则，将X移动到指定设备
            else:
                X = X.to(device)
            # 使用自动混合精度
            with torch.cuda.amp.autocast(enabled=True):
                # 提取特征
                embeddings = net.extract(X)
                # 将特征和权重进行线性变换，并归一化
                y += softmax(S * F.linear(F.normalize(embeddings), F.normalize(net.fc.weight))).cpu()
    # 将结果转换为numpy数组
    return np.array(list(Y.numpy() for Y in y))


def CLAHE_Convert(origin_input):
    # 创建CLAHE对象，clipLimit为对比度限制，tileGridSize为每个分块的大小
    clahe = cv.createCLAHE(clipLimit=40, tileGridSize=(10, 10))
    # 将输入图像转换为numpy数组
    t = np.asarray(origin_input)
    # 将图像从BGR颜色空间转换为HSV颜色空间
    t = cv.cvtColor(t, cv.COLOR_BGR2HSV)
    # 对HSV颜色空间中的V通道进行CLAHE处理
    t[:, :, -1] = clahe.apply(t[:, :, -1])
    # 将图像从HSV颜色空间转换回BGR颜色空间
    t = cv.cvtColor(t, cv.COLOR_HSV2BGR)
    # 将处理后的图像转换为PIL图像
    t = Img.fromarray(t)
    # 返回处理后的图像
    return t


# 定义训练集的转换
train_transform = transforms.Compose([
    CLAHE_Convert,  # CLAHE转换
    transforms.Resize(IMAGE_SIZE),  # 调整图像大小
    transforms.ColorJitter(brightness=0.2, contrast=0.05, saturation=0.1),  # 随机调整亮度、对比度和饱和度
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
    transforms.RandomApply(transforms=  # 随机应用以下变换
                           [transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.3, 0.4),
                                                         ratio=(1 / 3, 3), interpolation=  # 随机调整图像大小、比例和插值方式
                                                         transforms.InterpolationMode.BICUBIC)], p=0.2),  # 随机应用的概率
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 定义测试集和验证集的图像变换
val_test_transform = transforms.Compose([
    # 将图像转换为CLAHE增强后的图像
    CLAHE_Convert,
    # 将图像调整为指定大小
    transforms.Resize(IMAGE_SIZE),
    # 将图像转换为张量
    transforms.ToTensor(),
    # 对图像进行归一化处理
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

sorghum_test_dataset = Sorghum_Test_Dataset('D:\\Demo\\input\\test-csv\\test.csv', transform=val_test_transform)
sorghum_test_loader = DataLoader(sorghum_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True)

# 定义一个数据增强变换，包括CLAHE转换、调整图像大小、随机仿射变换、转换为张量、归一化
tta_transform0 = transforms.Compose([
    CLAHE_Convert,  # CLAHE转换
    transforms.Resize(IMAGE_SIZE),  # 调整图像大小
    transforms.RandomAffine(degrees=(0, 45), translate=(0.05, 0.1), scale=(0.95, 1)),  # 随机仿射变换
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 归一化

# 定义一个数据增强的变换，包括CLAHE转换、调整图像大小、随机垂直翻转、随机水平翻转、颜色抖动、转换为张量、归一化
tta_transform1 = transforms.Compose([
    CLAHE_Convert,  # CLAHE转换
    transforms.Resize(IMAGE_SIZE),  # 调整图像大小
    transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转，翻转概率为0.5
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，翻转概率为0.5
    transforms.ColorJitter(brightness=0.2, contrast=0.01, saturation=0.2),  # 颜色抖动，亮度变化范围为0.2，对比度变化范围为0.01，饱和度变化范围为0.2
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])  # 归一化，均值和标准差分别为[0.485, 0.456, 0.406]和[0.229, 0.224, 0.225]

# 定义一个数据增强的变换，包括CLAHE转换、调整大小、随机应用随机裁剪、转换为张量、归一化
tta_transform2 = transforms.Compose([
    CLAHE_Convert,  # CLAHE转换
    transforms.Resize(IMAGE_SIZE),  # 调整大小
    transforms.RandomApply(transforms=  # 随机应用
                           [transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.4, 0.5),  # 随机裁剪
                                                         ratio=(1 / 3, 3), interpolation=  # 裁剪比例和插值方式
                                                         transforms.InterpolationMode.BICUBIC)], p=0.2),  # 随机应用的概率
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 归一化

# 定义一个列表，包含三个变换函数
tta_transforms = [tta_transform0,
                  tta_transform1,
                  tta_transform2]

result_raw_original = np.load('D:\\Demo\\input\\test-csv\\test_result_raw_Original.npy')

result_raw_ttas = {'origin': result_raw_original, 'avg': result_raw_original}

result_raw_ttas['avg'] /= len(result_raw_ttas.keys()) - 1

result_ttas_sorted_val = {}
result_ttas_sorted_idx = {}

result_raw_ttas.keys()

# 遍历result_raw_ttas字典中的键值对
for key, val in result_raw_ttas.items():
    # 清空CUDA缓存
    torch.cuda.empty_cache()
    # 将val转换为torch.tensor类型，数据类型为float32，设备为cuda
    result_tta = torch.tensor(val, dtype=torch.float32, device='cuda')
    # 对result_tta进行排序，按照dim=1的维度，降序排列
    result_sorted_val, result_sorted_idx = result_tta.sort(dim=1, descending=True)
    # 将排序后的结果转换为numpy数组，并存储到result_ttas_sorted_val字典中
    result_ttas_sorted_val[key] = result_sorted_val.cpu().numpy()
    # 将排序后的索引转换为numpy数组，并存储到result_ttas_sorted_idx字典中
    result_ttas_sorted_idx[key] = result_sorted_idx.cpu().numpy()
    # 删除result_tta, result_sorted_val, result_sorted_idx变量
    del result_tta, result_sorted_val, result_sorted_idx

# 将result_ttas_sorted_val字典中的'origin'键对应的值转换为DataFrame
result_original_df = pd.DataFrame(result_ttas_sorted_val['origin'])

# 对result_original_df的前两列进行描述性统计，计算百分位数
result_original_df.iloc[:, [0, 1]].describe(percentiles=[0.05, 0.25, 0.35, 0.45, 0.65, 0.75, 0.95])

# 将result_ttas_sorted_val字典中的'avg'键对应的值转换为DataFrame
result_tta_avg_df = pd.DataFrame(result_ttas_sorted_val['avg'])

# 对result_tta_avg_df的前两列进行描述性统计，计算百分位数
result_tta_avg_df.iloc[:, [0, 1]].describe(percentiles=[0.05, 0.25, 0.35, 0.45, 0.65, 0.75, 0.95])

# 计算阈值，阈值为result_tta_avg_df的第一列的均值减去标准差的一半
Threshold = result_tta_avg_df.iloc[:, 0].mean() - result_tta_avg_df.iloc[:, 0].std() / 2  # trust interval [μ - σ/2, )

Threshold

# 获取result_ttas_sorted_val中'avg'列的形状
result_ttas_sorted_val['avg'].shape

# 将result_ttas_sorted_val中'avg'列赋值给result_sorted_val
result_sorted_val = result_ttas_sorted_val['avg']
# 将result_ttas_sorted_idx中'avg'列赋值给result_sorted_idx
result_sorted_idx = result_ttas_sorted_idx['avg']

# 获取result_sorted_val的长度
len(result_sorted_val)

# 读取sample_submission.csv文件
sub_file = pd.read_csv(os.path.join(root_in, 'sample_submission.csv'))

# 读取test.csv文件
result = pd.read_csv('D:\\Demo\\input\\test-csv\\test.csv')
# 计算result中image列的值与sub_file中filename列的值相等的数量
sum(result.image.map(lambda x: x.split('.jpeg')[0]) == sub_file.filename.map(lambda x: x.split('.png')[0]))

# 读取sample_submission.csv文件
result = pd.read_csv('D:\\Demo\\input\\dataset\\sample_submission.csv')
# 将result_sorted_idx中的值映射到result的cultivar列
result['cultivar'] = [labels_map.get(result_sorted_idx[i, 0]) for i in range(result_sorted_idx.shape[0])]
# 将result的索引设置为filename列
result = result.set_index('filename')
# 将result保存为submission.csv文件
result.to_csv(os.path.join(root_out, 'submission.csv'))
