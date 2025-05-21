import scipy.io as sio
import numpy as np
import cvxpy as cp
from cvxpy.atoms.elementwise.power import power
from util.read_data import read_data

def find_sparse_sol(Y, i, N, D):
    if i == 0:
        Ybari = Y[:, 1:N]
    if i == N - 1:
        Ybari = Y[:, 0:N - 1]
    if i != 0 and i != N - 1:
        Ybari = np.concatenate((Y[:, 0:i], Y[:, i + 1:N]), axis=1)
    yi = Y[:, i].reshape(D, 1)

    # this ci will contain the solution of the l1 optimisation problem:
    # min (||yi - Ybari*ci||F)^2 + lambda*||ci||1   st. sum(ci) = 1

    ci = cp.Variable(shape=(N - 1, 1))
    constraint = [cp.sum(ci) == 1]
    obj = cp.Minimize(power(cp.norm(yi - Ybari @ ci, 2), 2) + 199101 * cp.norm(ci, 1))  # lambda = 199101
    prob = cp.Problem(obj, constraint)
    prob.solve(solver='MOSEK')
    return ci.value

def extract_coefficient(X1, num_sample, dim_sample, save_path):
    C = np.concatenate((np.zeros((1, 1)), find_sparse_sol(X1, 0, num_sample, dim_sample)), axis=0)
    for i in range(1, num_sample):
        ci = find_sparse_sol(X1, i, num_sample, dim_sample)
        zero_element = np.zeros((1, 1))
        cif = np.concatenate((ci[0:i, :], zero_element, ci[i:num_sample, :]), axis=0)
        C = np.concatenate((C, cif), axis=1)
        print("iterator %d/%d" % (i, num_sample))
    np.savetxt(save_path, C, delimiter=",")
    print("The coefficient matrix is saved in " + save_path)


def main():
    import argparse
    import warnings
    parser = argparse.ArgumentParser(description='ExtractCofficient')
    parser.add_argument('--db', default='scene15',
                        choices=['20ngs', 'bbc_sports', 'digits', 'orl', 'scene15', 'yaleface', 'yaleb'])
    args = parser.parse_args()
    db = args.db
    if db == '20ngs':
        init_coef_file = '../datasets/20ngs/20newsgroups.mat'
        data = sio.loadmat(init_coef_file)
        # view1
        X1 = data['data'][0][0]
        # view2
        X2 = data['data'][0][1]
        # view3
        X3 = data['data'][0][2]
        num_sample = X1.shape[1]
        dim_sample = X1.shape[0]
        coef_path1 = 'datasets/20ngs/coef/coef1.csv'
        coef_path2 = 'datasets/20ngs/coef/coef2.csv'
        coef_path3 = 'datasets/20ngs/coef/coef3.csv'
        extract_coefficient(X1, num_sample, dim_sample, coef_path1)
        extract_coefficient(X2, num_sample, dim_sample, coef_path2)
        extract_coefficient(X3, num_sample, dim_sample, coef_path3)
    if db == 'orl':
        init_coef_file = '../datasets/orl/ORL.mat'
        data = sio.loadmat(init_coef_file)
        # view1
        X1 = data['x1']
        # view2
        X2 = data['x2']
        # view3
        X3 = data['x3']
        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        num_sample = X1.shape[1]
        dim_sample1 = X1.shape[0]
        dim_sample2 = X2.shape[0]
        dim_sample3 = X3.shape[0]
        coef_path1 = 'datasets/orl/coef/coef1.csv'
        coef_path2 = 'datasets/orl/coef/coef2.csv'
        coef_path3 = 'datasets/orl/coef/coef3.csv'
        extract_coefficient(X1, num_sample, dim_sample1, coef_path1)
        extract_coefficient(X2, num_sample, dim_sample2, coef_path2)
        extract_coefficient(X3, num_sample, dim_sample3, coef_path3)

    if db == 'yaleb':
        init_coef_file = '../datasets/yaleb/yaleB_mtv.mat'
        data = sio.loadmat(init_coef_file)
        # view1
        X1 = data['X'][0][0]
        # view2
        X2 = data['X'][0][1]
        # view3
        X3 = data['X'][0][2]
        num_sample = X1.shape[1]
        dim_sample1 = X1.shape[0]
        dim_sample2 = X2.shape[0]
        dim_sample3 = X3.shape[0]
        coef_path1 = 'datasets/yaleb/coef/coef1.csv'
        coef_path2 = 'datasets/yaleb/coef/coef2.csv'
        coef_path3 = 'datasets/yaleb/coef/coef3.csv'
        extract_coefficient(X1, num_sample, dim_sample1, coef_path1)
        extract_coefficient(X2, num_sample, dim_sample2, coef_path2)
        extract_coefficient(X3, num_sample, dim_sample3, coef_path3)

    if db == 'scene15':
        init_coef_file = '../datasets/scene15/scene-15.mat'
        data = sio.loadmat(init_coef_file)
        # view1, [4485 x 20]
        X1 = data['X'][0][0]
        # view2, [4485 x 59]
        X2 = data['X'][0][1]
        # view3, [4485 x 40]
        X3 = data['X'][0][2]
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        num_sample = X1.shape[1]
        dim_sample1 = X1.shape[0]
        dim_sample2 = X2.shape[0]
        dim_sample3 = X3.shape[0]
        coef_path1 = 'datasets/scene15/coef/coef1.csv'
        coef_path2 = 'datasets/scene15/coef/coef2.csv'
        coef_path3 = 'datasets/scene15/coef/coef3.csv'
        # extract_coefficient(X1, num_sample, dim_sample1, coef_path1)
        # extract_coefficient(X2, num_sample, dim_sample2, coef_path2)
        extract_coefficient(X3, num_sample, dim_sample3, coef_path3)


    if db == 'digits':
        init_coef_file = '../datasets/digits/uci-digit.mat'
        data = sio.loadmat(init_coef_file)
        # view1
        X1 = read_data('../datasets/digits/mfeat-fou')
        # view2
        X2 = read_data('../datasets/digits/mfeat-fac')
        # view3
        X3 = read_data('../datasets/digits/mfeat-kar')
        # view4
        X4 = read_data('../datasets/digits/mfeat-pix')
        # view5
        X5 = read_data('../datasets/digits/mfeat-zer')
        # view6
        X6 = read_data('../datasets/digits/mfeat-mor')
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        X4 = np.transpose(X4)
        X5 = np.transpose(X5)
        X6 = np.transpose(X6)
        num_sample = X1.shape[1]
        dim_sample1 = X1.shape[0]
        dim_sample2 = X2.shape[0]
        dim_sample3 = X3.shape[0]
        dim_sample4 = X4.shape[0]
        dim_sample5 = X5.shape[0]
        dim_sample6 = X6.shape[0]
        coef_path1 = 'datasets/digits/coef/coef1.csv'
        coef_path2 = 'datasets/digits/coef/coef2.csv'
        coef_path3 = 'datasets/digits/coef/coef3.csv'
        coef_path4 = 'datasets/digits/coef/coef4.csv'
        coef_path5 = 'datasets/digits/coef/coef5.csv'
        coef_path6 = 'datasets/digits/coef/coef6.csv'
        # extract_coefficient(X1, num_sample, dim_sample1, coef_path1)
        extract_coefficient(X2, num_sample, dim_sample2, coef_path2)
        # extract_coefficient(X3, num_sample, dim_sample3, coef_path3)
        # extract_coefficient(X4, num_sample, dim_sample4, coef_path4)
        # extract_coefficient(X5, num_sample, dim_sample5, coef_path5)
        # extract_coefficient(X6, num_sample, dim_sample6, coef_path6)

if __name__ == '__main__':
    main()