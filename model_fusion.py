# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 08:15:02 2023

@author: kkh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # matrix multiply
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias # 维度不一样 broadcast
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cuda:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device) # 单位矩阵

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten() # 压成一维向量
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class WordCnn(nn.Module):
    def __init__(self, word2vec_path, emb_dim=64, 
                 kernel_num=100, kernel_sizes=[3,4,5], dropout=0.5):
        super(WordCnn, self).__init__()
        dict = torch.load(word2vec_path)
        dict = torch.load(word2vec_path)
        dict  = torch.tanh(dict)
        dict_len, embed_size = dict.shape

        C = emb_dim
        Ci = 1 # in channnels
        Co = kernel_num # out channels
        Ks = kernel_sizes
        D = embed_size
        
        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.bn = nn.BatchNorm1d(2500)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks]) # (K, D) = Kernel_size
        # (N, Ci, Hi, Wi) -> (N, Co, Ho, Wo)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):

        x = self.lookup(x) # (1, word_lenth) -> (1, word_lenth=2500, embed_size=100)
    
        x = self.bn(x.float()).unsqueeze(1)  # (N, W, D) -> (N, Ci, W, D)   W : words_num       

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

    
class NoteNet(nn.Module):
    def __init__(self, vocab_size, word2vec_path,
                 emb_dim,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(NoteNet, self).__init__()
        self.device = device
        
        self.wordcnn = WordCnn(word2vec_path, emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim*2, batch_first=True)
        self.atten1 = nn.Linear(emb_dim*2, 1)
        self.feature1 = nn.Sequential(
           nn.ReLU(),
           nn.Linear(emb_dim*2, emb_dim))
        self.emb_dim = emb_dim


    def forward(self, input):
        f_seq = []
        
        for adm in input:
            text_index = torch.LongTensor(adm[3]).unsqueeze(0).to(self.device) # (1, word_lenth)
            feature_adm =self.wordcnn(text_index).unsqueeze(0) # (1, dim)
            f_seq.append(feature_adm)
        # print(f_seq)
        
        f_seq = torch.cat(f_seq, dim=1) # (1,seq,dim)  

        o1, h1 = self.gru(f_seq) # o1:(1, seq, dim*2) h1:(1,1,dim*2)

        atten1 = F.softmax(self.atten1(o1), dim=1) ##(1, seq, 1) 
        
        f1 = torch.sum(o1 * atten1, dim=1) # (1, dim*2) 
        
        feature_notes = self.feature1(f1) # (1, dim)    

        return feature_notes


class CodeNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, device):
        super(CodeNet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)]) # k为3 
        self.dropout = nn.Dropout(p = 0.4)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])
        """batch_first=True: (batch_size,seq,feature)"""
        self.atten1 = nn.Linear(emb_dim*2, 1)
        self.atten2 = nn.Linear(emb_dim*2, 1)
        self.feature1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim*4, emb_dim))
#        self.attention = nn.ModuleList([nn.Linear(emb_dim*2, 1) for _ in range(K-1)]) 

        self.init_weights()
        
    def forward(self, input):
        i1_seq = []
        i2_seq = []
        
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)
        for adm in input:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))

            i1_seq.append(i1)
            i2_seq.append(i2)
            
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)
        
        o1, h1 = self.encoders[0](
            i1_seq) # o1:(1, seq, dim*2) h1:(1,1,dim*2)
        o2, h2 = self.encoders[1](
            i2_seq)

        atten1 = F.softmax(self.atten1(o1), dim=1) ##(1, seq, 1) 
        atten2 = F.softmax(self.atten2(o2), dim=1) ##(1, seq, 1) 
        f1 = torch.sum(o1 * atten1, dim=1) # (1, dim*2)
        f2 = torch.sum(o2 * atten2, dim=1) # (1, dim*2)
        
        codes = torch.cat([f1, f2], dim=-1) # (1, dim*4)
        feature_codes = self.feature1(codes) # (1, dim)

        return feature_codes
        
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
            
            
class FusionNet(nn.Module):
    def __init__(self, vocab_size, ddi_adj, word2vec_path, embed_dim, device):
        super(FusionNet, self).__init__()
        D = embed_dim
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=embed_dim, adj=ddi_adj, device=device)
               
        self.code = CodeNet(vocab_size, embed_dim, device)
        self.note = NoteNet(vocab_size, word2vec_path, embed_dim, device)
        
        self.conv = nn.Conv2d(in_channels = 1, out_channels = D, kernel_size = (D+1, D+1)) ############################################        
        self.fci = nn.Linear(2*D, D) #####################################
        self.inter = nn.Linear(2*D, D)   
             
        self.fc1 = nn.Linear(2*D, 1)
        self.fc2 = nn.Linear(2*D, 1)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim * 3, vocab_size[2]))
        self.output1 = nn.Sequential( ###############################
            nn.ReLU(),
            nn.Linear(embed_dim * 2, vocab_size[2]))
        
    def forward(self, input):

        code = self.code(input) # 1*d
        note = self.note(input) # 1*d
        
        ddi = self.ddi_gcn().mean(dim=0).unsqueeze(dim=0) # (voc_size, dim) -> dim -> (1, dim)
        
        code_1 = torch.cat([code.squeeze(), torch.tensor([1]).to(self.device)], dim=-1) # d+1
        note_1 = torch.cat([note.squeeze(), torch.tensor([1]).to(self.device)], dim=-1)
        # x_e = torch.outer(code_1, note_1).unsqueeze(0) # (d+1, d+1) -> (1, d+1, d+1)
        x_e = torch.outer(code_1, note_1).unsqueeze(0).unsqueeze(0)
        # print(x_e.shape)
        x_e1 = self.conv(x_e)
        # print(x_e1.shape)
        x_e1 = x_e1.squeeze().unsqueeze(0)
        # x_e1 = self.conv(x_e).squeeze().unsqueeze(0)  # 1*2d*2d -> d*1*1 -> d -> 1*d
        x_i = torch.cat([code, note], dim=-1) 
        x_i1 = self.fci(x_i)  # 1*2d -> 1*d
        inter = self.inter(torch.cat([x_e1, x_i1], dim=-1))  # 1*2d -> 1*d
        
        w1 = torch.sigmoid(self.fc1(torch.cat([ddi, code], dim=-1)))
        w2 = torch.sigmoid(self.fc2(torch.cat([ddi, note], dim=-1)))
        
        output = self.output(torch.cat([w1*code, w2*note, inter], dim=-1))
        
        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()   # mul element-wise, mean求所有元素均值

            return output, batch_neg
        else:
            return output












