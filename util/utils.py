import os
import random
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
from munkres import Munkres


def best_map(L1, L2):
    # L1 should be the ground truth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    # 匈牙利分配算法
    m = Munkres()
    # 得到重新分配的标签的索引
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


# 计算模型结果的错误率
def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def next_batch(data, _index_in_epoch, batch_size, _epochs_completed):
    # 样本数
    _num_examples = data.shape[0]
    #
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        # label = label[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    return data[start:end], _index_in_epoch, _epochs_completed


def update_coef_by_label(coef, label):
    index = np.argsort(np.argsort(label))
    P_gt = np.eye(label.shape[0])
    P_gt = P_gt[:, index]
    label = P_gt @ label
    coef = P_gt @ coef @ P_gt.T
    return coef


def save_Coef(Coef, i, xlsx_dir):
    if not os.path.exists(xlsx_dir):
        os.makedirs(xlsx_dir)
    writer = pd.ExcelWriter(xlsx_dir + "W%d.xlsx" % (i))
    C = pd.DataFrame(Coef)
    C.to_excel(writer, 'page_1', float_format='%.8f')  # float_format 控制精度
    writer.save()


def save_Coef_mat(Coef, i, mat_dir):
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    # 构造要保存的数据字典
    data = {'coef': Coef}
    # 保存为.mat文件
    sio.savemat(os.path.join(mat_dir, "W%d.mat" % (i)), data)


def save_label(label,i,xlsx_dir):
    if not os.path.exists(xlsx_dir):
        os.makedirs(xlsx_dir)
    writer = pd.ExcelWriter(xlsx_dir + "W%d.xlsx" % (i))
    existing_data = pd.read_excel('./datasets/Colon/COAD_Survival.xlsx')
    # print(existing_data.shape)
    existing_data['Labels'] = label
    existing_data.to_excel(writer, index=False)
    writer.save()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True