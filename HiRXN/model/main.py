import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pandas as pd
import os 
import time 
import json 
from rdkit.Chem import Draw, AllChem

from HiRXN.model import HAN_model
from HiRXN.model import rxntokenizer
from HiRXN.model.process import load_variavle,save_variable


# config = argumentparser.ArgumentParser()
# torch.manual_seed(config.seed)
project_path = os.path.abspath(os.path.join(os.getcwd(), "HiRXN"))
data_path = os.path.join(project_path, "data_scaler")
check_path = os.path.join(project_path,'checkpoint')
vocab_path = os.path.join(project_path,'vocab')
vec_path = os.path.join(project_path,'group2vec')




# def get_pred_result(data):
#     # 生成测试结果
#     model.eval()
#     result = []
#     data = torch.tensor(data)
#
#     data = torch.unsqueeze(torch.unsqueeze(data, dim=0), dim=0)
#
#     if config.cuda and torch.cuda.is_available():
#
#         data = data.cuda()
#     else:
#         data = torch.autograd.Variable(data).long()
#     if config.cuda and torch.cuda.is_available():
#         out = model(data, gpu=True)
#     else:
#         out = model(data)
#     out.squeeze(1)
#     predict = out.cpu().detach().numpy()
#     if config.dataset =='uspto_1k':
#         predict = predict.astype(int)
#     result.extend(predict)
#     return result
def split_list(groups):
    arr = []
    index = groups.index('>')  # 获取标志值在数组中的索引
    array1 = groups[:index + 1]  # 获取索引之前的元素，即前一半
    array2 = groups[index + 1:]  # 获取索引及之后的元素，即后一半
    arr.append(array1)
    arr.append(array2)
    return arr
def regression(config):
    if config.cuda and torch.cuda.is_available():  # 是否使用gpu
        torch.cuda.set_device(config.gpu)

    if config.cuda and torch.cuda.is_available():  # 是否使用gpu
        torch.cuda.set_device(config.gpu)


    if config.dataset =='Buchwald-Hartwig':
        name = 'FullCV_09'
    if config.dataset =='suzuki':
        name = 'random_split_2'
    if config.dataset =='denmark':
        name = 'FullCV_04'
    if config.dataset =='uspto_1k':
        name =  'FullCV_01'
    # 导入训练集
    if config.dataset =='Buchwald-Hartwig':
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

    if config.dataset =='suzuki':
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

    if config.dataset =='denmark':
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

    # if config.dataset =='uspto_1k':
    #     train_df = pd.read_csv(data_path + '/uspto_1k_tpl/uspto_1k_train.csv')
    #     test_df = pd.read_csv(data_path + '/uspto_1k_tpl/uspto_1k_test.csv')
    #     train_df = train_df[['reaction', 'labels']]
    #     test_df = test_df[['reaction', 'labels']]
    #     train_df.columns = ['text', 'labels']
    #     test_df.columns = ['text', 'labels']

    ## Predictions
    # training_set = Hiero_Data(train_df, word2id=config.word2id, min_count=config.min_count,
    #                           max_sentence_length = config.max_sentence_length, batch_size=config.batch_size, is_pretrain=False, radius=int(radius),dataset=config.dataset)
    # training_iter = torch.utils.data.DataLoader(dataset=training_set,
    #                                             batch_size=config.batch_size,
    #                                             shuffle=False,
    #                                             num_workers=0)
    # # 导入测试集
    # test_set = Hiero_Data(test_df, min_count=config.min_count, word2id=training_set.word2id,
    #                       max_sentence_length = config.max_sentence_length, batch_size=config.batch_size, is_pretrain=True, radius=int(radius),dataset=config.dataset)
    # test_iter = torch.utils.data.DataLoader(dataset=test_set,
    #                                         batch_size=config.batch_size,
    #                                         shuffle=False,
    #                                         num_workers=0)

    if config.dataset == 'Buchwald-Hartwig':
        model_path = (check_path + '/BH/outputs_' + name + '.pt')
        word2id = load_variavle(vocab_path + '/Buchwald-Hartwig.pkl')
        weight = load_variavle(vec_path + '/BH_weight.pkl')
    if config.dataset == 'suzuki':
        model_path = (check_path + '/suzuki/suzuki_outputs_' + name + '.pt')
        word2id = load_variavle(vocab_path + '/suzuki.pkl')
        weight = load_variavle(vec_path + '/suzuki_weight.pkl')
    if config.dataset == 'denmark':
        model_path = (check_path + '/denmark/denmark_outputs_' + name + '.pt')
        word2id = load_variavle(vocab_path + '/denmark.pkl')
        weight = load_variavle(vec_path + '/denmark_weight.pkl')
    if config.dataset == 'uspto_1k':
        model_path = (check_path + '/uspto_1k/uspto_outputs_' + name + '.pt')
        word2id = load_variavle(vocab_path + '/uspto_1k.pkl')
        weight = load_variavle(vec_path + '/uspto_1k_weight.pkl')



    model = HAN_model.HAN_Model(word2id=word2id,vocab_size=len(word2id),
                                embedding_size=config.embedding_size,
                                gru_size=config.gru_size, class_num=config.class_num, weights=weight,
                                is_pretrain=True, drop_out_prob=0.1)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    # model.cuda()
    model.eval()
    

    # test = Hiero_Data(smiles, min_count=config.min_count, word2id=word2id,
    #                   max_sentence_length=config.max_sentence_length, batch_size=config.batch_size,
    #                   is_pretrain=True, radius=int(radius),dataset='Buchwald-Hartwig')
    '''
    提取token并整形
    '''
    datas = config.rxn_smiles
    if config.dataset == 'uspto_1k':
        splited = []
        r_1 = []
        w_1 = rxntokenizer(datas, config.radius)
        spl = split_list(w_1)
        for q in spl:
            for k, word in enumerate(q):
                try:
                    r_1.append(word2id[word])
                except:
                    pass
            splited.append(r_1)
            r_1 = []
        for j,groups in enumerate(spl):
            splited[j] = splited[j][0:config.max_sentence_length] + [word2id["<pad>"]] * (config.max_sentence_length - len(splited[j]))
        datas = splited
    else:
        a = []
        w_1 = rxntokenizer(datas, config.radius)
        for k, word in enumerate(w_1):
            try:
                a.append(word2id[word])
            except:
                a.append(word2id['<unk>'])
        datas = a

        datas = datas[0:config.max_sentence_length] + \
                [word2id["<pad>"]] * (config.max_sentence_length - len(datas))

    '''
    进行预测
    '''
    test_smiles = datas
    data = torch.tensor(test_smiles)
    # print(summary(model, input_size=data.shape))

    if config.dataset == 'uspto_1k':
        data = torch.unsqueeze(data, dim=0)
        # print(data.shape)
        # print(summary(model, input_size=data.shape))
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            out = model(data, gpu=True)
        else:
            data = torch.autograd.Variable(data).long()
            out = model(data)
        out = torch.argmax(out, 1)
        predict = out.cpu().detach().numpy()
        predict = predict.astype(int)
    else:
        data = torch.unsqueeze(torch.unsqueeze(data, dim=0), dim=0)
        # print(summary(model, input_size=data.shape))
        # data = torch.tensor(test_smiles)
        #
        # data = torch.unsqueeze(torch.unsqueeze(data, dim=0), dim=0)

        if config.cuda and torch.cuda.is_available():

            data = data.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        if config.cuda and torch.cuda.is_available():
            out = model(data, gpu=True)
        else:
            out = model(data)
        out.squeeze(1)
        predict = out.cpu().detach().numpy()


    y_preds=predict
    if config.dataset == 'Buchwald-Hartwig':
        y_preds = np.array(y_preds).flatten()
        y_preds = y_preds * std + mean
        pred = f"{y_preds.tolist()[0]:.3f}"

    if config.dataset == 'suzuki':
        y_preds = np.array(y_preds).flatten()
        y_preds = y_preds * std + mean
        y_preds = y_preds*100
        pred = f"{y_preds.tolist()[0]:.3f}"

    if config.dataset == 'denmark':
        y_preds = np.array(y_preds).flatten()
        y_preds = y_preds * std + mean
        pred = f"{y_preds.tolist()[0]:.3f}"

    if config.dataset == 'uspto_1k':
        y_preds = np.array(y_preds).flatten()
        pred = f"{y_preds.tolist()[0]}"

    rxn = AllChem.ReactionFromSmarts(config.rxn_smiles,useSmiles=True)
    rxn_img = Draw.ReactionToImage(rxn, useSVG=True)
    # static_dir = r"C:\Users\zhaox\Desktop\BDATOOL\app\public\reactions"
    rxn_filename = f"{config.task_id}_{time.strftime('%Y%m%d')}.svg"
    with open(os.path.join(config.static_dir, rxn_filename), 'w') as f:
        f.write(rxn_img)

    res = { # 将python对象编码成Json字符串
        "task_id" : f"{config.task_id}",
        "rxn_tokens" : list(set(w_1)),
        "prediction" : pred,
        "reaction_img_name": rxn_filename, 
    }
    with open(os.path.join(config.save_path, f"{config.task_id}_result.json"), 'w') as f:
        json.dump(res, f)
    return res 




