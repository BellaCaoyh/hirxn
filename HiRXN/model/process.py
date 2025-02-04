import copy
import sys

from gensim.models import KeyedVectors
from torch.utils import data
import os
import torch
from torch.utils import data
import os
import nltk
import numpy as np
import torch.nn as nn
import pickle
from collections import Counter
from tqdm import tqdm
import pandas as pd
from HiRXN.model import rxntokenizer
import gensim
# import config as argumentparser
# config = argumentparser.ArgumentParser()
import os
""" 数据集加载 """
up_dir_path = os.path.abspath(os.path.join(os.getcwd(), "Hiero"))
vocab_path = os.path.join(up_dir_path, "vocab")
vec_path = os.path.join(up_dir_path,'group2vec')


def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
   f=open(filename,'rb')
   r=pickle.load(f)
   f.close()
   return r


class Hiero_Data(data.DataLoader):
    def __init__(self,data, min_count,word2id=True,gibbs=False, max_sentence_length=100, batch_size=64, is_pretrain=False, radius=2,category = 'train',dataset = None):
        self.dataset = dataset
        if dataset==None:
            print('未选择数据集')
            sys.exit()
        self.path = os.path.abspath(".")
        self.is_pretrain = is_pretrain
        self.radius = radius
        self.gibbs = gibbs
        self.category = category
        self.data = data
        self.min_count = min_count
        self.word2id = word2id
        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        if isinstance(data,str):
            self.convert_data2id_single()
        else:
            self.datas, self.labels = self.Data_load()
            self.get_word2vec_2()
            for i in range(len(self.datas)):
                self.datas[i] = np.array(self.datas[i])

    def convert_data2id_single(self):
        datas = self.data
        if self.dataset == 'uspto_1k':
            r_1 = []
            spl = self.split_list(datas)
            for q in spl:
                for k, word in enumerate(q):
                    try:
                        r_1.append(self.word2id[word])
                    except:
                        # r_1.append(self.word2id['<unk>'])
                        pass
                datas.append(r_1)

                datas = datas[0:self.max_sentence_length] + \
                                [self.word2id["<pad>"]] * (self.max_sentence_length - len(datas))

        a = []
        w_1 = rxntokenizer(datas, self.radius)
        for k, word in enumerate(w_1):
            try:
                a.append(self.word2id[word])
            except:
                a.append(self.word2id['<unk>'])
        datas = a

        datas = datas[0:self.max_sentence_length] + \
                      [self.word2id["<pad>"]] * (self.max_sentence_length - len(datas))

        return datas

    def Data_load(self):
        '''
        读取文件，并生成word2id
        '''

        if self.category == 'train' and self.word2id == True:
            data_w2d = self.data
            # data_w2d = train_df
            words = list()
            for i in tqdm(range(data_w2d.shape[0])):
                row = list(data_w2d.loc[i, :])
                for text in row[0:]:
                    if type(text) == str:
                        w = rxntokenizer(text, self.radius)  # 提取化合物中的基团
                        words.append(w)
            self.word = words
            self.get_word2id(words)
        if self.category == 'train' and self.word2id == None:
            if self.dataset == 'Buchwald-Hartwig':
                self.word2id = load_variavle(vocab_path+'/Buchwald-Hartwig.pkl')
            if self.dataset == 'suzuki':
                self.word2id = load_variavle(vocab_path+'/suzuki.pkl')
            if self.dataset == 'denmark':
                self.word2id = load_variavle(vocab_path+'/denmark.pkl')
            if self.dataset == 'uspto_1k':
                self.word2id = load_variavle(vocab_path+'/uspto_1k_word.pkl')
        if self.category == 'test' and self.dataset == 'uspto_1k':
            self.word = load_variavle(vocab_path+'/uspto_1k_word.pkl')
            self.get_word2id(self.word)
        # 读取数据
        data = self.data
        datas = data['text']
        labels = data['labels'].values.tolist()
        np_data = [['text'] for i in range(len(datas))]
        for i, data in enumerate(datas):
            np_data[i][0] = data
        if self.dataset == 'uspto_1k':
            datas = self.convert_data2id_for_uspto(np_data)
        datas = self.convert_data2id(np_data)
        return datas, labels
    def split_list(self,groups):
        arr = []
        index = groups.index('>')  # 获取标志值在数组中的索引
        array1 = groups[:index+1]  # 获取索引之前的元素，即前一半
        array2 = groups[index+1:]  # 获取索引及之后的元素，即后一半
        arr.append(array1)
        arr.append(array2)
        return arr


    def get_word2id(self, datas):
        word_freq = {}
        for data in datas:
            for word in data:
                word_freq[word] = word_freq.get(word, 0) + 1
        word2id = {"<pad>": 0, "<unk>": 1}
        for word in word_freq:
            if word_freq[word] < self.min_count:
                continue
            else:
                word2id[word] = len(word2id)
        if self.dataset == 'Buchwald-Hartwig':
            save_variable(word2id, vocab_path + '/Buchwald-Hartwig.pkl')
        if self.dataset == 'suzuki':
            save_variable(word2id, vocab_path + '/suzuki.pkl')
        if self.dataset == 'denmark':
            save_variable(word2id, vocab_path + '/denmark.pkl')
        if self.dataset == 'uspto_1k':
            save_variable(word2id, vocab_path + '/uspto_1k.pkl')
        self.word2id = word2id

    # 将数据转化为id,句子必须一样的长度,每个文档的句子一样多
    def convert_data2id(self, datas):
        a = []
        for i, document in enumerate(datas):
            if i % 10000 == 0:
                print(i, len(datas))
            for j, sentence in enumerate(document):
                w_1 = rxntokenizer(sentence,self.radius)
                for k, word in enumerate(w_1):
                    try:
                        a.append(self.word2id[word])
                    except:
                        a.append(self.word2id['<unk>'])
                datas[i][j]=a
                a = []

                datas[i][j] = datas[i][j][0:self.max_sentence_length] + \
                              [self.word2id["<pad>"]] * (self.max_sentence_length - len(datas[i][j]))
        for i in range(0, len(datas), self.batch_size):
            max_data_length = max([len(x) for x in datas[i:i + self.batch_size]])
            for j in range(i, min(i + self.batch_size, len(datas))):
                datas[j] = datas[j] + [[self.word2id["<pad>"]] * self.max_sentence_length] * (
                            max_data_length - len(datas[j]))
        return datas

    def convert_data2id_for_uspto(self, datas):
        r_1 = []
        for i, reaction in enumerate(datas):
            if i % 10000 == 0:
                print(i, len(datas))
            reaction = copy.deepcopy(reaction)
            datas[i].clear()
            w_1 = rxntokenizer(reaction[0], self.radius)
            spl = self.split_list(w_1)
            for q in spl:
                for k, word in enumerate(q):
                    try:
                        r_1.append(self.word2id[word])
                    except:
                        # r_1.append(self.word2id['<unk>'])
                        pass
                datas[i].append(r_1)
                r_1 = []
            for j,groups in enumerate(spl):
                datas[i][j] = datas[i][j][0:self.max_sentence_length] + \
                          [self.word2id["<pad>"]] * (self.max_sentence_length - len(datas[i][j]))
                # datas[i][j][-1]=ph[i]##添加ph标志
        for i in range(0, len(datas), self.batch_size):
            max_data_length = max([len(x) for x in datas[i:i + self.batch_size]])
            for j in range(i, min(i + self.batch_size, len(datas))):
                datas[j] = datas[j] + [[self.word2id["<pad>"]] * self.max_sentence_length] * (
                            max_data_length - len(datas[j]))
        # finalldatas = [i for ii in datas for i in ii]
        return datas

    def get_word2vec(self):

        print("doing word2vec Embedding...")
        #
        vocab_size = len(self.word2id)
        embedding_dim = 200
        embedding = nn.Embedding(vocab_size, embedding_dim)
        word_indices = [self.word2id[word] for word in self.word2id.keys()]
        word_vectors = embedding(torch.LongTensor(word_indices))

        print(word_vectors)
        self.weight = word_vectors

    def get_word2vec_2(self):
        if not self.is_pretrain:
            model = gensim.models.word2vec.Word2Vec(sentences=self.word, vector_size=config.embedding_size,
                                                    max_vocab_size=len(self.word2id), workers=8, window=20, min_count=0,seed=0)
            if self.dataset == 'Buchwald-Hartwig':
                model.save(vec_path + '/BH_word2vec.pt')
            if self.dataset == 'suzuki':
                model.save(vec_path + '/Suzuki_word2vec.pt')
            if self.dataset == 'denmark':
                model.save(vec_path + '/denmark_word2vec.pt')
            if self.dataset == 'uspto_1k':
                model.save(vec_path + '/uspto_1k_word2vec.pt')
        '''
        保证测试集和训练集用同一套embedding
        '''
        if self.dataset == 'Buchwald-Hartwig':
            wvmodel = gensim.models.word2vec.Word2Vec.load(vec_path + '/BH_word2vec.pt')
        if self.dataset =='suzuki':
            wvmodel = gensim.models.word2vec.Word2Vec.load(vec_path + '/Suzuki_word2vec.pt')
        if self.dataset =='denmark':
            wvmodel = gensim.models.word2vec.Word2Vec.load(vec_path + '/denmark_word2vec.pt')
        if self.dataset =='uspto_1k':
            wvmodel = gensim.models.word2vec.Word2Vec.load(vec_path + '/uspto_1k_word2vec.pt')

        tmp = []
        for word, index in self.word2id.items():
            try:
                tmp.append(wvmodel.wv[word])
            except:
                pass
        mean = np.mean(np.array(tmp))
        std = np.std(np.array(tmp))
        print(mean, std)
        vocab_size = len(self.word2id)
        embed_size = 200
        np.random.seed(2)
        embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正太分布初始化方法
        for word, index in self.word2id.items():
            try:
                embedding_weights[index,:] = wvmodel.wv[word]
            except:
                pass
        self.weight = torch.from_numpy(embedding_weights).float()
        if self.dataset == 'Buchwald-Hartwig':
            save_variable(self.weight, vec_path + '/BH_weight.pkl')
        if self.dataset == 'suzuki':
            save_variable(self.weight, vec_path + '/suzuki_weight.pkl')
        if self.dataset == 'denmark':
            save_variable(self.weight, vec_path + '/denmark_weight.pkl')
        if self.dataset == 'uspto_1k':
            save_variable(self.weight, vec_path + '/uspto_1k_weight.pkl')



    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
    def get_token(self):
        tokens = rxntokenizer(self.data, self.radius)
        return tokens


if __name__ == "__main__":
    train_df = pd.read_csv('../HAN_model/data_scaler/Buchwald-Hartwig/random_split/' + 'FullCV_01' + '_train_temp_scaler.csv')


    train_df = train_df[['reaction', 'origin_output']]


    train_df.columns = ['text', 'labels']

    mean = train_df.labels.mean()
    std = train_df.labels.std()
    train_df['labels'] = (train_df['labels'] - mean) / std

    '''
    进行数据处理，提取word2id
    '''
    # if config.word2id == True:
    #    word2id = take_vacab(train_df)
    # if config.word2id == False:
    #    word2id = process.load_variavle('../data_scaler/Buchwald_Hartwig/BH_word2id.txt')
    '''
    模型处理数据
    '''

    training_set = Hiero_Data(train_df,word2id=True,min_count=0,
                              max_sentence_length=200, batch_size=64,
                              is_pretrain=True)
    training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=0)
    # a = imdb_data.__getitem__(1)

    # a = training_iter.dataset.datas
    # b = torch.Tensor(a)
    for datas, labels in training_iter:
        print(datas)

