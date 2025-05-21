import numpy as np
import os
import re
from csv import reader
os.path.join('./')
# def read_data(filename, pattern='0.[0-9]{8}'):
#     f = open(filename, 'r')
#     data = []
#     while True:
#         s = f.readline()
#         if not s:
#             break
#         s1 = re.findall(pattern, s)
#         data.append(s1)
#     return np.array(data)

def read_data(filename):
    f = open(filename, 'r', encoding='utf-8')
    data = []
    while True:
        s = f.readline()
        if not s:
            break
        s = s.strip()
        s1 = re.split('[\s]+', s)
        for item in s1:
            if (len(item)) == 0:
                s1.remove(item)
        s1 = [float(item) for item in s1]
        # print(s1)
        data.append(s1)
    return np.array(data)


def read_coef(file):
    '''数据预处理函数'''
    with open(file, 'r', encoding='utf-8') as f:
        '''数据按行读取'''
        data = list(reader(f))
    '''转化为numpy数组'''
    data = np.array(data)
    new_data = np.ones_like(data, dtype=float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            new_data[i][j] = float(data[i][j])
    # data = [float(item1) for item1 in item for item in data]
    return new_data

def read_Label(file):
    with open(file, 'r', encoding='utf-8') as f:
        '''数据按行读取'''
        data = list(reader(f))
    data = [int(item[0]) for item in data]
    data = np.array(data)
    return data
# data = read_data('../datasets/digits/mfeat-mor')
# print(data.shape)
# def main():
#     np.savetxt(save_path, C, delimiter=",")
# print(data)