import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
class HAN_Model(nn.Module):
    def __init__(self,vocab_size,embedding_size,gru_size,class_num,is_pretrain=False,weights=None,drop_out_prob=0.2,word2id=None):
        super(HAN_Model, self).__init__()
        self.word2id = word2id
        self.gru_size = gru_size
        self.dropout = nn.Dropout(p=drop_out_prob,inplace=True)
        if is_pretrain:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_gru = nn.GRU(input_size=embedding_size,hidden_size=gru_size,num_layers=2,
                               bidirectional=True,batch_first=True)#gibbs=num——layers==2
        #自己设的query Uw
        self.word_context = nn.Parameter(torch.Tensor(2*gru_size, 1),requires_grad=True)#将一个固定不可训练的tensor转换成可以训练的类型parameter
        # while torch.isnan(self.word_context).any():
        #     self.word_context = nn.Parameter(torch.Tensor(2 * gru_size, 1) * 0.5, requires_grad=True)
        self.word_dense = nn.Linear(2*gru_size,2*gru_size)

        self.sentence_gru = nn.GRU(input_size=2*gru_size,hidden_size=gru_size,num_layers=2,
                               bidirectional=True,batch_first=True)
        # 自己设的query Us
        self.sentence_context = nn.Parameter(torch.Tensor(2*gru_size, 1),requires_grad=True)
        self.sentence_dense = nn.Linear(2*gru_size,2*gru_size)
        self.fc = nn.Linear(2*gru_size,class_num)
        self.epoch_num = 0
        self.heat = dict()
        self.heat_2 = dict() 
    def forward(self, x,gpu=False):
        data = x
        sentence_num = x.shape[1]
        sentence_length = x.shape[2]
        x = x.view([-1,sentence_length])
        # 假设你的索引张量是index_tensor
        x = x.to(torch.int64)  # 转换为Long类型
        # x: bs*sentence_num*sentence_length -> (bs*sentence_num)*sentence_length 转成二维
        x_embedding = self.embedding(x) # (bs*sentence_num)*sentence_length*embedding_size
        word_outputs, word_hidden = self.word_gru(x_embedding) # word_outputs.shape: (bs*sentence_num)*sentence_length*2gru_size
        word_outputs_attention = torch.tanh(self.word_dense(word_outputs)) # (bs*sentence_num)*sentence_length*2gru_size
        # print(word_outputs_attention.shape)
        # print(word_outputs_attention)
        while torch.isnan(self.word_context).any():
            self.word_context = nn.Parameter(torch.Tensor(2*self.gru_size, 1).cuda(),requires_grad=True)
        weights = torch.matmul(word_outputs_attention,self.word_context) # (bs*sentence_num)*sentence_length*1
        '''
        word层级从中间分成两个向量分成两层
        '''
        weights = F.softmax(weights,dim=1) # (bs*sentence_num)*sentence_length*1
        # weight_word = weights
        x = x.unsqueeze(2) # (bs*sentence_num)*sentence_length*1 加维度
        # 权值矩阵：有值的地方保留，pad部分设为0
        if gpu:
            weights = torch.where(x!=0,weights,torch.full_like(x,0,dtype=torch.float).cuda())
        else:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float)) # (bs*sentence_num)*sentence_length*1
        # 和恢复为1
        weights = weights/(torch.sum(weights,dim=1).unsqueeze(1)+1e-4) # (bs*sentence_num)*sentence_length*1

        sentence_vector = torch.sum(word_outputs*weights,dim=1).view([-1,sentence_num,word_outputs.shape[-1]]) #bs*sentence_num*2gru_size
        sentence_outputs, sentence_hidden = self.sentence_gru(sentence_vector)# sentence_outputs.shape: bs*sentence_num*2gru_size
        attention_sentence_outputs = torch.tanh(self.sentence_dense(sentence_outputs)) # sentence_outputs.shape: bs*sentence_num*2gru_size
        # print(attention_sentence_outputs.shape)
        # print(attention_sentence_outputs)
        weights = torch.matmul(attention_sentence_outputs,self.sentence_context) # sentence_outputs.shape: bs*sentence_num*1
        weights = F.softmax(weights,dim=1) # sentence_outputs.shape: bs*sentence_num*1
        # weights = weights+weight_word
        x = x.view(-1, sentence_num, x.shape[1]) # bs*sentence_num*sentence_length
        x = torch.sum(x, dim=2).unsqueeze(2) # bs*sentence_num*1
        if gpu:
            weights = torch.where(x != 0, weights, torch.full_like(x,0,dtype=torch.float).cuda())
        else:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float)) #  bs*sentence_num*1
        weights = weights / (torch.sum(weights,dim=1).unsqueeze(1)+1e-4) # bs*sentence_num*1
        document_vector = torch.sum(sentence_outputs*weights,dim=1)# bs*2gru_size
        # document_vector = torch.cat((document_vector,ph_flag.view(-1,1)),dim=1)
        document_vector = self.dropout(document_vector)
        output = self.fc(document_vector) #bs*class_num
        return output
if __name__ == '__main__':
    han_model = HAN_Model(vocab_size=30000,embedding_size=200,gru_size=50,class_num=1)
    x = torch.Tensor(np.random.randint(low=2, high=11,size=(64,1,150))).long()
    # x = torch.Tensor(np.zeros([64,50,100])).long()
    A = np.infty
    A = torch.tensor(A)
    a = A-A
    print(a)
    b = torch.isnan(a).all()
    print(b)
    # x[0][0][0:10] = 1
    # print(x)
    output = han_model(x).cuda(0)
    print (output.shape)

