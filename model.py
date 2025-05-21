import torch
import torch.nn as nn
import torch.nn.functional as F
from util.choose_neighbor import choose_neighbor_coefficient


class CAE(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(CAE, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.encoder_layer1 = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.Sigmoid(),
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Linear(num_hidden, num_outputs),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_outputs, num_hidden),
            nn.Sigmoid(),
            nn.Linear(num_hidden, num_inputs),
            nn.Sigmoid(),
        )

    def forward(self, x):
        layer1 = self.encoder_layer1(x)
        layer2 = self.encoder_layer2(layer1)
        y = self.decoder(layer2)
        return y

    def loss_fn(self, x_recon, x):
        loss = F.mse_loss(x_recon, x, reduction='sum')
        return loss


class SelfExpression(nn.Module):
    def __init__(self, coe):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(torch.tensor(coe, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class DSRMSC(nn.Module):
    def __init__(self, num_sample, feat_size, num_hidden, num_outputs, batch_size, coe1, coe2, coe3):
        super(DSRMSC, self).__init__()
        self.n = num_sample
        self.coe1 = coe1
        self.coe2 = coe2
        self.coe3 = coe3
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.ae1 = CAE(feat_size[0], num_hidden[0], num_outputs[0])
        self.ae2 = CAE(feat_size[1], num_hidden[1], num_outputs[1])
        self.ae3 = CAE(feat_size[2], num_hidden[2], num_outputs[2])
        self.self_expression1 = SelfExpression(self.coe1)
        self.self_expression2 = SelfExpression(self.coe2)
        self.self_expression3 = SelfExpression(self.coe3)
        # self.Coefficient_consensus = nn.Parameter(torch.tensor((coe1 + coe2 + coe3) / 3, dtype=torch.float32), requires_grad=True)

    def forward(self, x1, x2, x3):
        # encoder
        v1_latent_pre = self.ae1.encoder_layer1(x1)
        v2_latent_pre = self.ae2.encoder_layer1(x2)
        v3_latent_pre = self.ae3.encoder_layer1(x3)
        v1_latent = self.ae1.encoder_layer2(v1_latent_pre)
        v2_latent = self.ae2.encoder_layer2(v2_latent_pre)
        v3_latent = self.ae3.encoder_layer2(v3_latent_pre)
        v_latent_pre_return = torch.stack([v1_latent_pre, v2_latent_pre, v3_latent_pre], dim=0)
        v_latent_return = torch.stack([v1_latent, v2_latent, v3_latent], dim=0)

        # 自表达层的输出
        v1_z_c_pre = self.self_expression1(v1_latent_pre)
        v2_z_c_pre = self.self_expression2(v2_latent_pre)
        v3_z_c_pre = self.self_expression3(v3_latent_pre)
        v1_z_c = self.self_expression1(v1_latent)
        v2_z_c = self.self_expression2(v2_latent)
        v3_z_c = self.self_expression3(v3_latent)
        v_z_c_pre_return = torch.stack([v1_z_c_pre, v2_z_c_pre, v3_z_c_pre], dim=0)
        v_z_c_return = torch.stack([v1_z_c, v2_z_c, v3_z_c], dim=0)

        # decoder
        x1_recon = self.ae1.decoder(v1_z_c)
        x2_recon = self.ae2.decoder(v2_z_c)
        x3_recon = self.ae3.decoder(v3_z_c)

        Coefficient_1 = self.self_expression1.Coefficient.cpu().detach()
        Coefficient_2 = self.self_expression2.Coefficient.cpu().detach()
        Coefficient_3 = self.self_expression3.Coefficient.cpu().detach()

        # 获取纯净比重A_C
        A1 = torch.tensor(choose_neighbor_coefficient(Coefficient_1, 8, 20)).cuda()
        A2 = torch.tensor(choose_neighbor_coefficient(Coefficient_2, 8, 20)).cuda()
        A3 = torch.tensor(choose_neighbor_coefficient(Coefficient_3, 8, 20)).cuda()

        pure_1 = torch.mul(A1, self.self_expression1.Coefficient)
        pure_2 = torch.mul(A2, self.self_expression2.Coefficient)
        pure_3 = torch.mul(A3, self.self_expression3.Coefficient)

        consensus = ( pure_1 + pure_2 + pure_3 )/3

        return consensus, pure_1 , pure_2 , pure_3 , x1_recon, x2_recon, x3_recon, v_latent_pre_return, v_latent_return, v_z_c_pre_return, v_z_c_return

    def loss_fn(self,
                consensus,
                pure_1,
                pure_2,
                pure_3,
                x1,
                x2,
                x3,
                x1_recon,
                x2_recon,
                x3_recon,
                v_latent_pre,
                v_latent,
                v_z_c_pre,
                v_z_c,
                weight_selfExp,
                weight_diff,
                weight_coef):
        # 重构损失
        loss_ae1 = F.mse_loss(x1_recon, x1, reduction='sum')
        loss_ae2 = F.mse_loss(x2_recon, x2, reduction='sum')
        loss_ae3 = F.mse_loss(x3_recon, x3, reduction='sum')
        loss_ae = (loss_ae1 + loss_ae2 + loss_ae3) / 3

        # 自表达损失
        loss_selfExp_v_pre, loss_selfExp_v = 0, 0
        for i in range(v_latent_pre.shape[0]):
            loss_selfExp_v_pre = loss_selfExp_v_pre + F.mse_loss(v_latent_pre[i], v_z_c_pre[i], reduction='sum')
        for i in range(v_latent.shape[0]):
            loss_selfExp_v = loss_selfExp_v + F.mse_loss(v_latent[i], v_z_c[i], reduction='sum')

        # loss_selfExp = v1_pre自表达损失 + v2_pre自表达损失 + v3_pre自表达损失 + v1自表达损失 + v2自表达损失 + v3自表达损失
        loss_selfExp = (loss_selfExp_v_pre + loss_selfExp_v) / v_latent_pre.shape[0]

        # 亲和矩阵正则化
        loss_coef = torch.sum(torch.pow(consensus, 2))

        # 共识损失
        with torch.no_grad():
            w_wight1 = 1 / (2 * F.mse_loss(pure_1, consensus, reduction='sum').sqrt())
            w_wight2 = 1 / (2 * F.mse_loss(pure_2, consensus, reduction='sum').sqrt())
            w_wight3 = 1 / (2 * F.mse_loss(pure_3, consensus, reduction='sum').sqrt())

        loss_diff_coe1 = w_wight1 * F.mse_loss(pure_1, consensus, reduction='sum')
        loss_diff_coe2 = w_wight2 * F.mse_loss(pure_2, consensus, reduction='sum')
        loss_diff_coe3 = w_wight3 * F.mse_loss(pure_3, consensus, reduction='sum')

        loss_diff_coe = (loss_diff_coe1 + loss_diff_coe2 + loss_diff_coe3) / 3

        # 模型总的损失 = 重构损失 + 自表达损失 + 亲和矩阵正则化 + 共识损失
        loss = loss_ae + weight_selfExp * loss_selfExp + weight_diff * loss_diff_coe + weight_coef * loss_coef
        return loss

