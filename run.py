import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import scipy.io as sio

from model import CAE, DSRMSC
from util.plot_graph import show_heat_map
from util.read_data import read_coef
from util.utils import next_batch, save_Coef_mat, save_label, update_coef_by_label, set_random_seed
from clustering import spectral_clustering, acc, nmi, post_proC, thrC

import warnings
warnings.simplefilter('ignore')


def pre_train(model,
              data,
              lr,
              batch_size,
              save_path,
              device=torch.device('cuda:0')):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    it = 0
    display_step = 500
    save_step = 5000
    _index_in_epoch = 0
    _epochs = 0

    max_num = 80001
    while it < max_num:
        batch_x, _index_in_epoch, _epochs = next_batch(data, _index_in_epoch, batch_size, _epochs)
        batch_x = np.reshape(batch_x, [batch_size, data.shape[1]])
        if not isinstance(batch_x, torch.Tensor):
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=device)
        batch_x = batch_x.to(device)
        batch_x_recon = model(batch_x)
        loss = model.loss_fn(batch_x_recon, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        it = it + 1
        avg_cost = loss / (batch_size)
        if it % display_step == 0:
            print("cost: %.8f" % avg_cost)
        if it % save_step == 0:
            torch.save(model.state_dict(), save_path)
            print('model saved in file: %s' % save_path)
    return


def train(args,
          model,
          x1,
          x2,
          x3,
          epochs,
          lr=1e-3,
          weight_selfExp=1.0,
          weight_diff=1.0,
          weight_coef=1.0,
          device=torch.device('cuda:0'),
          show=10,
          alpha=0.6):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x1, torch.Tensor):
        x1 = torch.tensor(x1, dtype=torch.float32, device=device)
    x1 = x1.to(device)
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, dtype=torch.float32, device=device)
    x2 = x2.to(device)
    if not isinstance(x3, torch.Tensor):
        x3 = torch.tensor(x3, dtype=torch.float32, device=device)
    x3 = x3.to(device)
    print(f"weight_selfExp: {weight_selfExp}, weight_diff: {weight_diff}, weight_coef: {weight_coef}")

    print("Start Training ...")
    for epoch in range(epochs):

        # model.to(device)
        consensus, pure_1, pure_2, pure_3,x1_recon, x2_recon, x3_recon, v_latent_pre, v_latent, v_z_c_pre, v_z_c = model(x1, x2, x3)
        loss = model.loss_fn(consensus, pure_1 , pure_2 , pure_3 ,x1, x2, x3, x1_recon, x2_recon, x3_recon, v_latent_pre, v_latent, v_z_c_pre, v_z_c,
                             weight_selfExp, weight_diff, weight_coef)

        # grap1 = model.self_expression1.Coefficient.detach().to('cpu').numpy()
        # grap2 = model.self_expression2.Coefficient.detach().to('cpu').numpy()
        # grap3 = model.self_expression3.Coefficient.detach().to('cpu').numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Coef = consensus.detach().cpu().numpy()

        if (epoch == 0) or (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}: loss: {loss.item() / x1.shape[0]}")

        if (epoch % show == 0 or epoch == epochs - 1):
            coef1 = thrC(Coef, alpha=alpha)
            y_pred, L = post_proC(coef1, args.n_class, 5, 1)

            # xlsx_dir3_label = "./DSRMSC_W/" + args.db + "/label1" + "/"
            # save_label(y_pred, epoch / show, xlsx_dir3_label)

            # new_coef = update_coef_by_label(coef1, y_pred)
            # show_heat_map(new_coef, title='Colon_epoch_%d' % (epoch))
            # plt.show()

            # xlsx_dir1 = "./DSRMSC_W/" + args.db + "/1" + "/"
            # save_Coef_mat(Coef, epoch / show, xlsx_dir1)
    pred_name = f"./save/{args.db}/"
    if not os.path.exists(pred_name):
        os.makedirs(pred_name)
    sio.savemat(f"{pred_name}/{args.db}_cls{args.n_class}_pred.mat", {"pred": y_pred})
    print(f"The final predicted labels are saved as {pred_name}/{args.db}_cls{args.n_class}_pred.mat.")


def args_parser():
    parser = argparse.ArgumentParser(description='DSRMSC')
    parser.add_argument('--db', default='Colon',
                        choices=['Breast', 'Colon', 'Glioblastoma', 'Kidney', 'Lung',
                                 'BIC', 'COAD', 'GBM', 'SARC', 'OV', 'SKCM', 'LIHC',
                                 'AML', 'KIRC', 'LUSC'])
    parser.add_argument('--n_class', default=3, type=int)
    parser.add_argument('--show_freq', default=1, type=int)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results')
    parser.add_argument('--is_pretrain', default=False)
    return parser.parse_args()


def main():
    datasets = {'Breast', 'Colon', 'Glioblastoma', 'Kidney', 'Lung',
                'BIC', 'COAD', 'GBM','SARC', 'OV', 'SKCM', 'LIHC',
                'AML', 'KIRC', 'LUSC'}
    args = args_parser()
    args.db = "Colon"
    # args.n_class = 3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    db = args.db
    if db == 'Glioblastoma':
        init_coef_file = './datasets/Glioblastoma/GBM1.mat'
        dataset = sio.loadmat(init_coef_file)
        print(init_coef_file)
        print(db)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        # label
        lr = 1e-3
        epochs = 100
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        print("Gene:", X1.shape)
        print("Methy:", X2.shape)
        print("Mirna:", X3.shape)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'Colon':
        init_coef_file = './datasets/Colon/Colon.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        # X1 = np.transpose(X1.astype(np.float32))
        # X2 = np.transpose(X2.astype(np.float32))
        # X3 = np.transpose(X3.astype(np.float32))
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'Lung':
        init_coef_file = './datasets/Lung/LSCC1.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'Kidney':
        init_coef_file = './datasets/Kidney/KRCCC1.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'Breast':
        init_coef_file = './datasets/Breast/BIC1.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3,lr=lr, epochs=epochs, device=device)
    if db == 'BIC':
        init_coef_file = './datasets/BIC/BIC.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'GBM':
        init_coef_file = './datasets/GBM/GBM.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'COAD':
        init_coef_file = './datasets/COAD/COAD.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'LUSC':
        init_coef_file = './datasets/LUSC/LUSC.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'SKCM':
        init_coef_file = './datasets/SKCM/SKCM.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'LIHC':
        init_coef_file = './datasets/LIHC/LIHC.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'KIRC':
        init_coef_file = './datasets/KIRC/KIRC.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'OV':
        init_coef_file = './datasets/OV/OV.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'AML':
        init_coef_file = './datasets/AML/AML.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)
    if db == 'SARC':
        init_coef_file = './datasets/SARC/SARC.mat'
        dataset = sio.loadmat(init_coef_file)
        # view1
        X1 = dataset['Gene']
        X2 = dataset['Methy']
        X3 = dataset['Mirna']
        X1 = np.log1p(X1)
        X3 = np.log1p(X3)
        # label
        lr = 1e-3
        epochs = 1000
        X1 = np.transpose(X1)
        X2 = np.transpose(X2)
        X3 = np.transpose(X3)
        train_and_save_model(args, X1, X2, X3, lr=lr, epochs=epochs, device=device)


def train_and_save_model(args,
                         X1,
                         X2,
                         X3,
                         weight_selfExp=0.01,
                         weight_coef=0.01,
                         weight_diff=0.01,
                         lr=1e-3,
                         epochs=10000,
                         device=torch.device('cuda:0')):
    print(f"Training on {device} ...")
    num_hidden = [128, 128, 128]
    num_outputs =[128, 128, 128]
    num_sample = X1.shape[0]
    batch_size = num_sample
    feat_size = torch.tensor(np.array([X1.shape[1], X2.shape[1], X3.shape[1]]), dtype=torch.int64)

    pretrain_batchsize = int(num_sample / 5.0)
    model_path1 = f'datasets/{args.db}/view1/view1.pkl'
    model_path2 = f'datasets/{args.db}/view2/view2.pkl'
    model_path3 = f'datasets/{args.db}/view3/view3.pkl'
    coef_path1 = f'datasets/{args.db}/coef/coef111.csv'
    coef_path2 = f'datasets/{args.db}/coef/coef222.csv'
    coef_path3 = f'datasets/{args.db}/coef/coef333.csv'
    coe1 = read_coef(coef_path1)
    coe2 = read_coef(coef_path2)
    coe3 = read_coef(coef_path3)

    coef1 = thrC(coe1, alpha=0.5)          #对系数矩阵coef进行 阈值化处理，减少噪音
    y_pred, L = post_proC(coef1, 3, 5, 1)
    new_coef = update_coef_by_label(coef1, y_pred)
    show_heat_map(new_coef, title=f"{args.db}_epoch_init")

    pretrain1 = args.is_pretrain
    pretrain2 = args.is_pretrain
    pretrain3 = args.is_pretrain

    if pretrain1:
        ae1 = CAE(X1.shape[1], num_hidden[0], num_outputs[0])
        ae1.to(device)
        pre_train(ae1, X1, lr, pretrain_batchsize, model_path1, device)
    if pretrain2:
        ae2 = CAE(X2.shape[1], num_hidden[1], num_outputs[1])
        ae2.to(device)
        pre_train(ae2, X2, lr, pretrain_batchsize, model_path2, device)
    if pretrain3:
        ae3 = CAE(X3.shape[1], num_hidden[2], num_outputs[2])
        ae3.to(device)
        pre_train(ae3, X3, lr, pretrain_batchsize, model_path3, device)

    dsrmsc = DSRMSC(num_sample, feat_size, num_hidden, num_outputs, batch_size, coe1, coe2, coe3)
    dsrmsc.to(device)
    ae1_state_dict = torch.load(model_path1)
    dsrmsc.ae1.load_state_dict(ae1_state_dict)
    ae2_state_dict = torch.load(model_path2)
    dsrmsc.ae2.load_state_dict(ae2_state_dict)
    ae3_state_dict = torch.load(model_path3)
    dsrmsc.ae3.load_state_dict(ae3_state_dict)
    print("Successfully load pretrained AE weights.")
    train(args,dsrmsc, X1, X2, X3, epochs, lr, weight_selfExp, weight_coef, weight_diff, device, show=args.show_freq)
    torch.save(dsrmsc.state_dict(), args.save_dir + '/%s-model.ckp' % args.db)
    print(f"Training completed. The model is saved as {args.save_dir}/{args.db}_model.ckp.")


if __name__ == '__main__':
    set_random_seed(100)
    main()
