import os

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def show_heat_map(data, title=None, max=None, min=None, save_path=None, cmap='viridis', step=50):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.abs(data)
    if max is not None:
        data[data > max] = max
    if min is not None:
        data[(data > 0) & (data < min)] = min
    if title is not None:
        font={
            'family': 'Times New Roman',
            'size': 25
        }
        plt.title(title, font=font)
    x_ticks = np.arange(0, data.shape[0] + 1, step)
    # ax = sns.heatmap(data, xticklabels=x_ticks, cmap='Blues_r', vmin=0, vmax=1)#cmap='viridis',
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(x_ticks)
    # ax.set_xticklabels(ax.get_xticks())
    # ax.set_yticklabels(ax.get_yticks())
    sns.heatmap(data, cmap=cmap)
    # sns.heatmap(data, cmap='Blues_r')
    # sns.heatmap(data, cmap='Blues')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def show_heat_map1(data, title=None, max=None, min=None, save_path=None):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.abs(data)
    if max is not None:
        data[data > max] = max
    if min is not None:
        data[(data > 0) & (data < min)] = min
    if title is not None:
        plt.title(title)
    # sns.heatmap(data)
    plt.matshow(data)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


color = ['green', 'orange', 'paleturquoise', 'grey', 'purple', 'chocolate', 'pink', 'red', 'blue', 'black']
def t_sne(data, label, title=None, save_path=None):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(data)
    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(10, 10))
    if title is not None:
        plt.title(title)
    plt.rcParams.update({'font.size': 20})
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[color[item] for item in label], label="t-SNE")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, label="t-SNE")
    # plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def t_sne_new(data: list, label, cols=2, title=None, save_path=None):
    nums = len(data)
    if title is not None:
        assert isinstance(title, list) and len(title) == nums
    data_tsne = []
    # Processing data
    for i in range(nums):
        if isinstance(data[i], torch.Tensor):
            data[i] = data[i].detach().cpu().numpy()
        data_tsne.append(TSNE(n_components=2, random_state=33).fit_transform(data[i]))
    #
    plt.figure(figsize=(10 * nums, 10))
    # plt.rcParams.update({'font.size': 20})
    rows = int(nums / cols)
    for i in range(nums):
        # subplots
        plt.subplot(rows, cols, i + 1)
        plt.scatter(data_tsne[i][:, 0], data_tsne[i][:, 1], c=label, label="fea")#c=[color[int(item)] for item in label]
        if title is not None:
            plt.title(title[i], fontdict={'fontsize': 14})
        plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_history(history, title):
    plt.title(title)
    plt.plot(history)
    plt.show()