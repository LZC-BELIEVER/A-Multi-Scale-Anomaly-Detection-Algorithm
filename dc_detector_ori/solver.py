import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.DCdetector import DCdetector
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
warnings.filterwarnings('ignore')
import random

#defined_pool_sizes = [2,3]

def set_seed(seed):
    random.seed(seed)  # Python的随机种子
    np.random.seed(seed)  # Numpy的随机种子
    torch.manual_seed(seed)  # PyTorch的CPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch的GPU随机种子
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 避免CuDNN自动寻找最佳卷积算法

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):    
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        dataset_name = os.path.basename(self.dataset).replace('./', '')
        checkpoint_path = os.path.join(path, f'{dataset_name}_checkpoint.pth')
        #checkpoint_path = os.path.join(path, str(self.dataset) + '_checkpoint.pth')
        print(f"Saving checkpoint to: {checkpoint_path}")  # 打印实际的保存路径
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        set_seed(42)
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset,)
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)

        print(f"len_thre{len(self.thre_loader)}")
        print(f"len_test{len(self.test_loader)}")
        print(f"len_vali{len(self.vali_loader)}")
        print(f"len_train{len(self.train_loader)}")
        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        

    def build_model(self):
        self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size, channel=self.input_c)
        
        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            #series, prior = self.model(input)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)


    def train(self):
        time_now = time.time()  # 获取当前时间
        path = self.model_save_path  # 获取模型保存路径
        if not os.path.exists(path):  # 如果路径不存在
            os.makedirs(path)  # 创建路径
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_path)  # 初始化早停对象
        train_steps = len(self.train_loader)  # 获取训练步骤数

        for epoch in range(self.num_epochs):  # 遍历所有 epochs
            iter_count = 0  # 初始化迭代计数
            epoch_time = time.time()  # 获取 epoch 开始时间
            self.model.train()  # 设置模型为训练模式
            for i, (input_data, labels) in enumerate(self.train_loader):  # 遍历训练数据加载器
                self.optimizer.zero_grad()  # 清零梯度
                iter_count += 1  # 增加迭代计数
                input = input_data.float().to(self.device)  # 将输入数据转换为 float 并移动到设备上
                # series, prior = self.model(input)  # 获取模型输出***
                series, prior = self.model(input)

                series_loss = 0.0  # 初始化 series 损失
                prior_loss = 0.0  # 初始化 prior 损失

                for u in range(len(prior)):  # 遍历 prior
                    series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach())) + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach(), series[u])))  # 计算 series 损失
                    prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)))))  # 计算 prior 损失

                loss = prior_loss - series_loss  # 计算总损失

                if (i + 1) % 100 == 0:  # 每 100 次迭代
                    speed = (time.time() - time_now) / iter_count  # 计算每次迭代的速度
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)  # 计算剩余时间
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))  # 打印速度和剩余时间
                    iter_count = 0  # 重置迭代计数
                    time_now = time.time()  # 更新当前时间

                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新优化器

            vali_loss1, vali_loss2 = self.vali(self.test_loader)  # 验证损失

            print("Epoch: {0}, Cost time: {1:.3f}s ".format(epoch + 1, time.time() - epoch_time))  # 打印 epoch 和时间消耗
            early_stopping(vali_loss1, vali_loss2, self.model, path)  # 调用早停
            if early_stopping.early_stop:  # 如果提前停止
                break  # 退出训练循环
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)  # 调整学习率

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))  # 加载模型检查点
        self.model.eval()  # 设置模型为评估模式
        temperature = 50  # 设置温度参数

        
        # (1) 训练集统计
        attens_energy = []  # 初始化能量列表
        for i, (input_data, labels) in enumerate(self.train_loader):  # 遍历训练数据加载器
            input = input_data.float().to(self.device)  # 将输入数据转换为 float 并移动到设备上
            #series, prior = self.model(input)  # 获取模型输出
            series, prior = self.model(input)  # 获取模型输出
            series_loss = 0.0  # 初始化 series 损失
            prior_loss = 0.0  # 初始化 prior 损失
            for u in range(len(prior)):  # 遍历 prior
                if u == 0:  # 如果是第一个 prior
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) * temperature  # 计算 series 损失
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature  # 计算 prior 损失
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) * temperature  # 累加 series 损失
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature  # 累加 prior 损失

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)  # 计算 softmax 作为度量
            cri = metric.detach().cpu().numpy()  # 转换为 NumPy 数组
            attens_energy.append(cri)  # 添加到能量列表
        
        print("this is tranatt")
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)  # 连接能量列表并重塑
        train_energy = np.array(attens_energy)  # 转换为 NumPy 数组

        # (2) 寻找阈值
        attens_energy = []  # 初始化能量列表
        for i, (input_data, labels) in enumerate(self.thre_loader):  # 遍历阈值数据加载器
            input = input_data.float().to(self.device)  # 将输入数据转换为 float 并移动到设备上
            series, prior = self.model(input)  # 获取模型输出
            #series, prior = self.model(input)
            series_loss = 0.0  # 初始化 series 损失
            prior_loss = 0.0  # 初始化 prior 损失
            for u in range(len(prior)):  # 遍历 prior
                if u == 0:  # 如果是第一个 prior
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) * temperature  # 计算 series 损失
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature  # 计算 prior 损失
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) * temperature  # 累加 series 损失
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature  # 累加 prior 损失

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)  # 计算 softmax 作为度量
            cri = metric.detach().cpu().numpy()  # 转换为 NumPy 数组
            attens_energy.append(cri)  # 添加到能量列表

        print("len_att")
        print(len(attens_energy))
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)  # 连接能量列表并重塑
        test_energy = np.array(attens_energy)  # 转换为 NumPy 数组
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)  # 合并训练和测试能量
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)  # 计算阈值
        print(f"!!!combined_energy:{combined_energy}")
        print(f"thresh:{thresh}")
        print("Threshold :", thresh)  # 打印阈值
        np.savetxt('thresh.txt', combined_energy, delimiter='\n', fmt='%f')

        # (3) 在测试集上进行评估
        test_labels = []  # 初始化测试标签列表
        attens_energy = []  # 初始化能量列表``
        count=0
        for i, (input_data, labels) in enumerate(self.thre_loader):  # 遍历阈值数据加载器
            input = input_data.float().to(self.device)  # 将输入数据转换为 float 并移动到设备上
            #series, prior = self.model(input)  # 获取模型输出
            series, prior = self.model(input)
            series_loss = 0.0  # 初始化 series 损失
            prior_loss = 0.0  # 初始化 prior 损失
            for u in range(len(prior)):  # 遍历 prior
                if u == 0:  # 如果是第一个 prior
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) * temperature  # 计算 series 损失
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature  # 计算 prior 损失
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) * temperature  # 累加 series 损失
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature  # 累加 prior 损失
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)  # 计算 softmax 作为度量
            #print(f"-series_loss:{-series_loss}")
            #print(f"- prior_loss:{- prior_loss}")
            #print(f"metric:{metric}")
            cri = metric.detach().cpu().numpy()  # 转换为 NumPy 数组
            #print(f"cri:{cri}")
            attens_energy.append(cri)  # 添加到能量列表
            test_labels.append(labels)  # 添加到测试标签列表
            #print(f"attens_energy:{attens_energy}")
            count=count+1

        print(count)
        #print(f"len_attens_energy:{len(attens_energy)}")
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)  # 连接能量列表并重塑
        #print(f"attens_energy:{attens_energy}")
        np.savetxt('attens_energy.txt', attens_energy, delimiter='\n', fmt='%f')
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)  # 连接测试标签列表并重塑
        test_energy = np.array(attens_energy)  # 转换为 NumPy 数组
        test_labels = np.array(test_labels)  # 转换为 NumPy 数组

        
        np.savetxt('test_energy.txt', test_energy, delimiter='\n', fmt='%f')
        pred = (test_energy > thresh).astype(int)  # 计算预测标签
        #print(f"pred_aftertest:{pred}")
        np.savetxt('pred_aftertest.txt', pred, delimiter='\n', fmt='%f')
        gt = test_labels.astype(int)  # 转换真实标签为整数

        matrix = [self.index]  # 初始化结果矩阵
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)  # 计算评估指标
        for key, value in scores_simple.items():  # 遍历评估指标
            matrix.append(value)  # 将指标添加到结果矩阵
            print('{0:21} : {1:0.4f}'.format(key, value))  # 打印评估指标

        anomaly_state = False  # 初始化异常状态
        np.savetxt('gt.txt', gt, delimiter='\n', fmt='%f')
        for i in range(len(gt)):  # 遍历真实标签
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:  # 如果真实标签为 1 且预测标签为 1，且当前不是异常状态
                anomaly_state = True  # 设置异常状态
                for j in range(i, 0, -1):  # 从当前位置向前遍历
                    if gt[j] == 0:  # 如果真实标签为 0
                        break  # 退出循环
                    else:
                        if pred[j] == 0:  # 如果预测标签为 0
                            pred[j] = 1  # 将预测标签设置为 1
                for j in range(i, len(gt)):  # 从当前位置向后遍历
                    if gt[j] == 0:  # 如果真实标签为 0
                        break  # 退出循环
                    else:
                        if pred[j] == 0:  # 如果预测标签为 0
                            pred[j] = 1  # 将预测标签设置为 1
            elif gt[i] == 0:  # 如果真实标签为 0
                anomaly_state = False  # 取消异常状态
            if anomaly_state:  # 如果当前是异常状态
                pred[i] = 1  # 将预测标签设置为 1

        pred = np.array(pred)  # 转换为 NumPy 数组
        gt = np.array(gt)  # 转换为 NumPy 数组
        pred_count=0
        for k in range(0,len(pred)):
            if (pred[k]==1):
                pred_count=pred_count+1

        from sklearn.metrics import precision_recall_fscore_support  # 导入精确度、召回率和 F 分数支持
        from sklearn.metrics import accuracy_score  # 导入准确度评分

        accuracy = accuracy_score(gt, pred)  # 计算准确度
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')  # 计算精确度、召回率和 F 分数
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))  # 打印评估结果
        with open('./result.txt', 'a') as f:
            line="Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} \n".format(accuracy, precision, recall, f_score)
            f.write(line)
        with open('./result_count.txt', 'a') as f:
            line="{:d}\n".format(pred_count)
            f.write(line)

        if self.data_path == 'UCR' or 'UCR_AUG':  # 如果数据路径是 UCR 或 UCR_AUG
            import csv  # 导入 CSV 模块
            with open('result/'+self.data_path+'.csv', 'a+') as f:  # 以追加模式打开 CSV 文件
                writer = csv.writer(f)  # 创建 CSV 写入器
                writer.writerow(matrix)  # 将结果矩阵写入 CSV 文件

        return accuracy, precision, recall, f_score  # 返回评估结果