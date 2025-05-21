import scipy.io as sio
import torch
import numpy as np

def choose_neighbor_coefficient(Z, num_good, num_ordinary):
    # num_good: 8 好邻居的个数
    # num_ordinary: 20 普通邻居
    # the optimal parameter set varies with different datasets

    # 对角线设置为0
    Z = Z - torch.diag(torch.diag(Z))

    # 归一化
    z = torch.sum(Z, dim=1)
    w = torch.zeros_like(Z)
    for i in range(Z.shape[0]):
        w[i, :] = Z[i, :] / z[i]

    # 首先选取num_ordinary个普通邻居
    C = torch.zeros_like(w)
    S_number_temp = torch.zeros((w.shape[0], num_ordinary))
    S_weight_temp = torch.zeros((w.shape[0], num_ordinary))
    for i in range(w.shape[0]):
        C[i, :] = w[i, :] + w[:, i].t()
        for j in range(num_ordinary):
            p, q = torch.max(C[i, :], dim=0)
            S_number_temp[i, j] = q.item()+1
            S_weight_temp[i, j] = p.item()
            C[i, q] = 0

    # 再选取num_good个好邻居
    C = torch.zeros_like(w)
    num = 0
    S_weight = torch.zeros_like(w)
    S_number = torch.zeros((w.shape[0], num_good))
    for i in range(w.shape[0]):
        C[i, :] = w[i, :] + w[:, i].t()
        for j in range(num_good):
            p, q = torch.max(C[i, :], dim=0)

            for k in range(w.shape[0]):
                if (S_number_temp[q, :] == i+1).any():
                    S_weight[i, q] = p
                    S_number[i, j] = q.item()+1
                    C[i, q] = 0
                    break
                else:
                    num += 1
                    C[i, q] = 0
                    p, q = torch.max(C[i, :], dim=0)


    # 将好邻居设置为1，不够的再补够
    for k in range(w.shape[0]):
        # idx = np.where(S_number[k, :] == 0)
        idx = (S_number[k, :] == 0).nonzero(as_tuple=True)[0]
        S_number[k, idx] = k
        S_weight[k, idx] = 1

    # 最后进行对称
    S_weight = (S_weight + S_weight.t()) / 2

    # return S_weight.numpy()
    return S_weight

#
# mat = sio.loadmat('matlab.mat')
# Z = 'your_variable_name'
# var_name = 'Z'
# array = np.array(mat[var_name])
# Z = torch.from_numpy(array)
#
# Z = torch.arange(0, 100).reshape(10, 10).t()
# Z = Z.float()
# num_good = 3
# num_ordinary = 6
# A = choose_neighbor_coefficient(Z, num_good, num_ordinary)
# print(A)
