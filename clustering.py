import numpy as np
import torch
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

from util.utils import best_map


nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)


def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, ro):
    """C: coefficient matrix, K: number of clusters, d: dimension of each subspace"""
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L


def form_structure_matrix(y_pred, num_classes):
    # 去除y_pred中大小为1的维度
    y_pred = torch.squeeze(y_pred)
    # numel(): 返回数组中元素的个数
    Q = torch.zeros((y_pred.numel(), num_classes))
    for i in range(y_pred.numel()):
        j = y_pred[i]
        Q[i, j] = 1

    return Q


def form_Theta(Q):
    Theta = torch.zeros((Q.shape[0], Q.shape[0]))
    for i in range(Q.shape[0]):
        # repeat([a, b]) 在第一个维度上复制a-1次，在第二个维度上复制b-1次
        Qq = Q[i].repeat([Q.shape[0],1])
        # 与Q[i]相同的类别值为0
        Theta[i, :] = 1/2*torch.sum(torch.pow((Q - Qq),2), 1)
    return Theta


# 计算模型结果的准确率
def acc_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return 1 - missrate


def spectral_clustering(C, K, d, alpha, ro):

    C = thrC(C, alpha)
    C = np.abs(C)
    y, _ = post_proC(C, K, d, ro)
    return y, C


def kmeans(data, K):
    kmeans = cluster.KMeans(n_clusters=K, random_state=10)
    kmeans.fit(data)
    grp = kmeans.fit_predict(data)
    return grp


def get_center(feature, K):
    kmeans = cluster.KMeans(n_clusters=K, random_state=10)
    kmeans.fit(feature)
    center = kmeans.cluster_centers_
    return center
