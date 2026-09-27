import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import time
import datetime
import os
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter

# 导入自定义模块（请确保这些文件在你的路径中）
from utils import train_epoch, valid_epoch, mkdirs, Logger, record_output
from model import Net
from config import config, DefaultConfigs

# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

# --- 1. 数据加载与预处理模块 ---

def loadData():
    """根据配置文件加载对应的高光谱数据集"""
    dataset_path = '../../dataset/'
    if config.dataset == 'PU':
        data = sio.loadmat(dataset_path + 'PaviaU.mat')['paviaU']
        labels = sio.loadmat(dataset_path + 'PaviaU_gt.mat')['paviaU_gt']
    elif config.dataset == 'IP':
        data = sio.loadmat(dataset_path + 'Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat(dataset_path + 'Indian_pines_gt.mat')['indian_pines_gt']
    elif config.dataset == 'SA':
        data = sio.loadmat(dataset_path + 'Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat(dataset_path + 'Salinas_gt.mat')['salinas_gt']
    elif config.dataset == 'HST':
        data = sio.loadmat(dataset_path + 'houston/houston_hsi.mat')['houston_hsi']
        labels = sio.loadmat(dataset_path + 'houston/houston_gt.mat')['gt']
    elif config.dataset == 'LK':
        data = sio.loadmat(dataset_path + 'WHU/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        labels = sio.loadmat(dataset_path + 'WHU/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt'] 
    return data, labels

def applyPCA(X, numComponents):
    """对原始高光谱数据进行PCA降维"""
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def padWithZeros(X, margin=2):
    """在空间维度进行零填充，以处理边界Patch"""
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX

def createImageCubes(X, y, windowSize, removeZeroLabels=True):
    """将全图数据切割成固定大小的 Patch (Cubic)"""
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex += 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1  # 标签转为从0开始
    return patchesData, patchesLabels

# --- 2. 数据划分与 Loader 模块 ---

def splitTrainTestSet_useRatio(X, y, testRatio, randomState=6):
    """按照比例划分训练集和测试集，并保证每类至少有3个样本"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, 
                                                        random_state=randomState, stratify=y)
    min_samples_train = 3
    train_counter = Counter(y_train)
    for label in set(y):
        if train_counter[label] < min_samples_train:
            test_indices = [i for i, label_test in enumerate(y_test) if label_test == label]
            needed = min_samples_train - train_counter[label]
            move_indices = test_indices[:needed]
            X_train = np.concatenate([X_train, X_test[move_indices]])
            y_train = np.concatenate([y_train, y_test[move_indices]])
            X_test = np.delete(X_test, move_indices, axis=0)
            y_test = np.delete(y_test, move_indices, axis=0)
    return X_train, X_test, y_train, y_test

class HsiDataset(torch.utils.data.Dataset):
    """通用高光谱数据集包装类"""
    def __init__(self, X, y):
        self.len = X.shape[0]
        self.x_data = torch.FloatTensor(X)
        self.y_data = torch.LongTensor(y)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

def create_data_loader():
    """主数据处理流程：加载 -> PCA -> 切块 -> 划分 -> Loader"""
    X, y = loadData()
    config.num_class = int(np.max(y))
    config.num_band = X.shape[2]

    # PCA 降维
    if config.use_pca:
        X_pca = applyPCA(X, numComponents=config.pca_components)
    else:
        X_pca = X
    
    # 标准化
    X_pca = X_pca.reshape(-1, X_pca.shape[-1])
    X_pca = preprocessing.StandardScaler().fit_transform(X_pca)
    X_pca = X_pca.reshape(X.shape[0], X.shape[1], -1)

    # 切块
    X_cubes, y_labels = createImageCubes(X_pca, y, windowSize=config.patch_size)
    
    # 划分 (此处默认使用 Ratio 模式)
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet_useRatio(X_cubes, y_labels, config.test_ratio, config.seed)
    
    # 调整维度为 (B, C, H, W) 或 (B, H, W, C) 取决于你的 Net 结构，这里保持与原代码一致
    Xtrain = Xtrain.reshape(-1, config.patch_size, config.patch_size, config.pca_components if config.use_pca else config.num_band)
    Xtest = Xtest.reshape(-1, config.patch_size, config.patch_size, config.pca_components if config.use_pca else config.num_band)
    X_all = X_cubes.reshape(-1, config.patch_size, config.patch_size, config.pca_components if config.use_pca else config.num_band)

    # 构建 DataLoader
    train_loader = Data.DataLoader(HsiDataset(Xtrain, ytrain), batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(HsiDataset(Xtest, ytest), batch_size=config.batch_size, shuffle=False)
    all_loader = Data.DataLoader(HsiDataset(X_all, y_labels), batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader, all_loader, y_labels

# --- 3. 评价指标计算 ---

def output_metric(tar, pre):
    """计算 OA, AA, Kappa 以及各类别精度"""
    matrix = confusion_matrix(tar, pre)
    number = np.trace(matrix)
    sum_rows = np.sum(matrix, axis=1)
    sum_cols = np.sum(matrix, axis=0)
    
    OA = number / np.sum(matrix)
    AA = matrix.diagonal() / sum_rows
    AA_mean = np.mean(AA)
    
    pe = np.sum(sum_rows * sum_cols) / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

# --- 4. 训练核心逻辑 ---

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False

def train_1times(log, run_index):
    """单次完整训练与测试流程"""
    setup_seed(run_index)
    train_loader, test_loader, all_loader, _ = create_data_loader()

    model = Net(num_class=config.num_class, embed_dim=config.embed_dim, num_band=config.num_band).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.epoches // 10, gamma=config.gamma)

    BestAcc = 0
    counter = 0 
    print(f"Start training for Run {run_index}")

    for epoch in range(config.epoches):
        model.train()
        _, _, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer)
        OA_t, _, _, _ = output_metric(tar_t, pre_t)
        
        scheduler.step()

        # 验证
        if (epoch % config.valid_freq == 0) or (epoch == config.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, test_loader, criterion)
            OA_v, AA_v, Kappa_v, CA_v = output_metric(tar_v, pre_v)
            
            print(f"Epoch: {epoch+1:03d} | Train OA: {OA_t:.4f} | Valid OA: {OA_v:.4f}")

            if OA_v > BestAcc:
                torch.save(model.state_dict(), f'./Wights/Best_{config.dataset}_Net.pkl')
                BestAcc = OA_v
                counter = 0
            else:
                counter += 1
                if counter >= config.patience:
                    log.write(f"Early stop at epoch {epoch+1}")
                    break

    # 测试最优模型
    model.load_state_dict(torch.load(f'./Wights/Best_{config.dataset}_Net.pkl'))
    model.eval()
    
    tic = time.perf_counter()
    tar_test, pre_test = valid_epoch(model, test_loader, criterion)
    toc = time.perf_counter()
    
    OA, AA, Kappa, CA = output_metric(tar_test, pre_test)
    
    # 记录结果
    log.write(f"Run {run_index} Final Records: OA: {OA:.4f} | AA: {AA:.4f} | Kappa: {Kappa:.4f}")
    log.write(f"Test Time: {toc - tic:.2f}s\n")

    return OA, AA, Kappa, CA, (toc - tic)

# --- 5. 程序入口 ---

if __name__ == '__main__':
    # 初始化环境
    mkdirs(config.best_model_folder, config.logs_folder, config.records_folder)
    log = Logger()
    log.open(os.path.join(config.logs_folder, f"{config.dataset.upper()}_log.txt"), mode='w')

    OA_all, AA_all, KAPPA_all, CA_all, TIME_all = [], [], [], [], []

    # 循环运行多次实验
    for i in range(config.run_times):
        log.write(f"{'='*30} Start Run {i} {'='*30}")
        oa, aa, kappa, ca, runtime = train_1times(log, i)
        
        OA_all.append(oa)
        AA_all.append(aa)
        KAPPA_all.append(kappa)
        CA_all.append(ca)
        TIME_all.append(runtime)
        log.write(f"{'='*30} End Run {i} {'='*30}\n")

    # 汇总并保存最终统计报表
    current_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
    report_name = f"records/{current_time}_{config.dataset}_final_report.txt"
    
    record_output(OA_all, AA_all, KAPPA_all, CA_all, TIME_all, report_name)
    
    log.write("All runs completed. Final results saved to records folder.")
    log.close()