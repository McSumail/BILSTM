import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
#



class BILSTM(nn.Module):
    def __init__(self, **kwargs):
        super(BILSTM, self).__init__()
        self.args = kwargs
        self.hidden_dim = kwargs["lstm_hidden_dim"]
        self.num_layers = kwargs["lstm_num_layers"]
        self.batch_size=kwargs["batch_size"]
        self.max_sent_len=kwargs["max_sent_len"]
        # 总的词汇量
        self.vocab_size = kwargs["vocab_size"]
        self.word_dim = kwargs["word_dim"]
        self.class_size = kwargs["class_size"]
        self.dropout=kwargs["dropout"]
        self.wv_matrix = kwargs["wv_matrix"]
        self.word_embedding=kwargs["word_embedding"]
        # one for UNK and one for zero padding
        # self.embed = nn.Embedding(V, D, max_norm=config.max_norm)
        self.embed = nn.Embedding(self.vocab_size+2, self.word_dim, padding_idx=self.vocab_size+1)
        # pretrained  embedding
        if self.word_embedding:
            self.embed.weight.data.copy_(torch.from_numpy(self.wv_matrix))
        self.bilstm = nn.LSTM(self.word_dim, self.hidden_dim // 2, num_layers=1, dropout=self.dropout, bidirectional=True,
                              bias=False)
        # print(self.bilstm)

        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, self.class_size)
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        embed = self.embed(x)
        x = embed.view( embed.size(1),len(x), -1)
        bilstm_out, _ = self.bilstm(x)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        # bilstm_out = F.dropout(bilstm_out, p=self.dropout, training=self.training)
        y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(y)
        logit = y
        return logit

