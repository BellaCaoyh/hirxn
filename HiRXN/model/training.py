'''
Author: Caoyh
Date: 2024-03-04 19:27:30
LastEditors: BellaCaoyh caoyh_cyh@163.com
LastEditTime: 2024-11-11 20:26:41
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_log_error,accuracy_score,confusion_matrix
import os
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd
# config = argumentparser.ArgumentParser()

from .HAN_model import HAN_Model
from .process import Hiero_Data


def train(config):
    torch.manual_seed(config.seed)
    if config.cuda and torch.cuda.is_available():  # 是否使用gpu
        torch.cuda.set_device(config.gpu)
    # up_dir_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    up_dir_path = 'HiRXN'
    data_path = os.path.join(up_dir_path, "data_scaler")
    check_path = os.path.join(up_dir_path,'checkpoint')
    vocab_path = os.path.join(up_dir_path,'vocab')
    vec_path = os.path.join(up_dir_path,'group2vec')

    if config.dataset =='Buchwald-Hartwig':
        NAME_SPLIT = [
            'FullCV_08',
        ]
        # NAME_SPLIT = [
        #     'FullCV_09'
        # ]
    if config.dataset =='suzuki':
        NAME_SPLIT = [
            'random_split_0', 'random_split_1', 'random_split_2', 'random_split_3',
            'random_split_4', 'random_split_5','random_split_6','random_split_7',
            'random_split_8', 'random_split_9'
        ]
    if config.dataset =='denmark':
        NAME_SPLIT = [
            'FullCV_01', 'FullCV_02', 'FullCV_03', 'FullCV_04',
            'FullCV_05', 'FullCV_06','FullCV_07','FullCV_08',
            'FullCV_09', 'FullCV_10'
        ]
    if config.dataset =='uspto_1k':
        NAME_SPLIT = [
            'FullCV_01'
        ]


    def get_test_result(data_iter):
        # 生成测试结果
        model.eval()
        acc = 0
        num = 0
        for data, label in data_iter:
            if config.cuda and torch.cuda.is_available():
                data = data.cuda()
                if config.dataset =='uspto_1k':
                    label = label.to(torch.float32)
                else:
                    label = label.to(torch.float32)
                label = label.cuda()
            else:
                data = torch.autograd.Variable(data).float()
            if config.cuda and torch.cuda.is_available():
                out = model(data, gpu=True)
            else:
                out = model(data)
            if config.dataset == 'uspto_1k':
                out = torch.argmax(out, 1)
            else:
                out.squeeze(1)
            predict = out.cpu().detach().numpy()
            truth = label.cpu().detach().numpy()

            if config.dataset =='uspto_1k':
                predict = predict.astype(int)
                num += len(predict)
                # acc = confusion_matrix(truth,predict,labels =[i for i in range(1001)] )
                acc += accuracy_score(truth, predict,normalize=None,sample_weight=None)
            else:
                num += 1
                acc += r2_score(truth,predict)
        accurency = acc/num
        return accurency
    # 导入训练集
    count = 0
    for name in NAME_SPLIT:
        if config.dataset == 'Buchwald-Hartwig':
            train_df = pd.read_csv(
                data_path + '/Buchwald-Hartwig/random_split/' + name + '_train_temp_scaler.csv')
            test_df = pd.read_csv(
                data_path + '/Buchwald-Hartwig/random_split/' + name + '_test_temp_scaler.csv')
            train_df = train_df[['reaction', 'origin_output']]
            test_df = test_df[['reaction', 'origin_output']]
            train_df.columns = ['text', 'labels']
            test_df.columns = ['text', 'labels']
            mean = train_df.labels.mean()
            std = train_df.labels.std()
            train_df['labels'] = (train_df['labels'] - mean) / std
            test_df['labels'] = (test_df['labels'] - mean) / std

        if config.dataset == 'suzuki':
            train_df = pd.read_csv(data_path + '/Suzuki/temp/' + name + '_train_temp_scaler.csv')
            test_df = pd.read_csv(data_path + '/Suzuki/temp/' + name + '_test_temp_scaler.csv')
            train_df = train_df[['reaction', 'origin_output']]
            test_df = test_df[['reaction', 'origin_output']]
            train_df.columns = ['text', 'labels']
            test_df.columns = ['text', 'labels']
            mean = train_df.labels.mean()
            std = train_df.labels.std()
            train_df['labels'] = (train_df['labels'] - mean) / std
            test_df['labels'] = (test_df['labels'] - mean) / std

        if config.dataset == 'denmark':
            train_df = pd.read_csv(data_path + '/denmark_0414/' + name + '_train_products_generated_temp.csv')
            test_df = pd.read_csv(data_path + '/denmark_0414/' + name + '_test_products_generated_temp.csv')
            train_df = train_df[['reaction', 'Output']]
            test_df = test_df[['reaction', 'Output']]
            train_df.columns = ['text', 'labels']
            test_df.columns = ['text', 'labels']
            mean = train_df.labels.mean()
            std = train_df.labels.std()
            train_df['labels'] = (train_df['labels'] - mean) / std
            test_df['labels'] = (test_df['labels'] - mean) / std

        if config.dataset =='uspto_1k':
            train_df = pd.read_csv(data_path + '/uspto_1k/uspto_1k_train.csv')
            test_df = pd.read_csv(data_path + '/uspto_1k/uspto_1k_test.csv')
            train_df = train_df[['reaction', 'labels']]
            test_df = test_df[['reaction', 'labels']]
            train_df.columns = ['text', 'labels']
            test_df.columns = ['text', 'labels']



        '''
        模型处理数据
        '''

        training_set = Hiero_Data(train_df, min_count=config.min_count, word2id=config.word2id,
                                  max_sentence_length=config.max_sentence_length, batch_size=config.batch_size,
                                  is_pretrain=False,dataset=config.dataset,radius=config.radius)
        training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                                    batch_size=config.batch_size,
                                                    shuffle=False,
                                                    num_workers=0)
        # 导入测试集
        test_set = Hiero_Data(test_df, min_count=config.min_count, word2id=training_set.word2id,
                              max_sentence_length=config.max_sentence_length, batch_size=config.batch_size,
                              is_pretrain=True,dataset=config.dataset,radius=config.radius)
        test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=config.batch_size,
                                                shuffle=False,
                                                num_workers=0)

        if config.cuda and torch.cuda.is_available():
            training_set.weight = training_set.weight.cuda()
        model = HAN_Model(vocab_size=len(training_set.word2id),
                                    embedding_size=config.embedding_size,
                                    gru_size=config.gru_size, class_num=config.class_num, weights=training_set.weight,
                                    is_pretrain=True, drop_out_prob=config.drop_out,word2id=training_set.word2id)

        if config.cuda and torch.cuda.is_available():  # 如果使用gpu，将模型送进gpu
            model.cuda()
        if config.class_num ==1 :#定义回归任务损失函数
            criterion = nn.SmoothL1Loss()
        else:#定义分类任务损失函数
            criterion = nn.CrossEntropyLoss()# 这里会做softmax
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        # lr_scheduler =optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.1,patience=10,verbose=True,threshold=0.001)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,200],gamma=0.5,verbose=True)
        loss = -1
        loss_flag = []
        acc_flag = []
        best_acc = 0
        for epoch in range(config.epoch):
            model.train()
            process_bar = tqdm(training_iter)
            for data, label in process_bar:
                if config.cuda and torch.cuda.is_available():
                    data = data.cuda()
                    if config.dataset == 'uspto_1k':
                        pass
                    else:
                        label = label.to(torch.float32)
                    label = label.cuda()
                else:
                    data = torch.autograd.Variable(data).float()
                label = torch.autograd.Variable(label).squeeze()
                if config.cuda and torch.cuda.is_available():
                    out = model(data, gpu=True)
                else:
                    out = model(data)
                out = out.squeeze(1)

                if config.dataset == 'uspto_1k':

                    loss_now = criterion(out, autograd.Variable(label.long()))
                else:
                    loss_now = criterion(out, autograd.Variable(label))

                if loss == -1:
                    loss = loss_now.data.item()
                else:
                    loss = 0.95 * loss + 0.05 * loss_now.data.item()  # gibbs
                    # loss = 0.15 * loss + 0.85 * loss_now.data.item() #yields
                loss_flag.append(loss)
                process_bar.set_postfix(loss=loss_now.data.item())
                process_bar.update()
                optimizer.zero_grad()
                loss_now.backward()
                optimizer.step()
            test_acc = get_test_result(test_iter)
            # lr_scheduler.step()
            acc_flag.append(test_acc)
            # 保存精度最好状态下的模型
            if epoch > 10 and test_acc > best_acc:
                best_acc = test_acc
                if config.dataset =='Buchwald-Hartwig':
                    torch.save(model,check_path + '/BH/outputs_' + name + 'new.pt')
                if config.dataset =='suzuki':
                    torch.save(model,check_path + '/suzuki/suzuki_outputs_' + name + 'new.pt')
                if config.dataset =='denmark':
                    torch.save(model,check_path + '/denmark/denmark_outputs_' + name + 'new.pt')
                if config.dataset =='uspto_1k':
                    # torch.save(model.state_dict(), '../HAN_model/checkpoint/uspto_1k_tpl/uspto_outputs_' + name+'_'+str(config.learning_rate)+'_'+str(config.max_sentence_length) +'_'+str(config.drop_out)+'_'+str(config.gru_size)+str(config.min_count) + '.pt')
                    torch.save(model,check_path + '/uspto_1k/uspto_outputs_' + name +'new.pt')

            print("epoch:" + str(epoch), "    The test acc is: %.5f" % test_acc)
        fianll_acc = get_test_result(test_iter)
        print("最终最好测试结果的r2分数：%.5f" % best_acc,'NAME:'+name)
        plt.figure(1)
        plt.plot(range(len(loss_flag)), loss_flag)
        plt.savefig('../img/loss'+name+'.png')
        plt.figure(2)
        plt.plot(range(len(acc_flag)), acc_flag)
        plt.savefig('../img/acc'+name+'.png')
        plt.show()
        count+=1