# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:57:17 2022

@author: kkh
"""

import torch
import argparse
import numpy as np
import dill
import time
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
import random

from model_fusion import FusionNet             #################################
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'model_fusion'                 #################################


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument("--word2vec_path", type=str, default="../../data/dict_vectors_1204.pt")

args = parser.parse_args()
model_name = args.model_name
word2vec_path = args.word2vec_path

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict) # 键不存在时返回值{}
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input): #################################

            target_output1 = model(input[:adm_idx+1])
        
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp)) # 排序 默认升序
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
        multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    """这两行验证时候可以取消了"""
    # dill.dump(obj=smm_record, file=open('../data/codenet_records.pkl', 'wb'))
    # dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))

    print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    data_path = '../../data/records_index_1204.pkl'
    voc_path = '../../data/voc_final_1103.pkl' # the vocabulary list to transform medical word to corresponding idx

    ddi_adj_path = '../../data/ddi_A_final.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    
    random.seed(1203)
    random.shuffle(data)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 40
    LR = 0.0002
    TARGET_DDI = 0.06
    DIM = 128

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word)) # idx2word为列表，word2idx为字典
    ##################################
    model = FusionNet(voc_size, ddi_adj, word2vec_path, embed_dim=DIM, device=device)

    model.to(device=device)

    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=LR)


    history = defaultdict(list)
    for epoch in range(EPOCH):
        loss_record1 = []
        start_time = time.time()
        model.train()
        prediction_loss_cnt = 0
        neg_loss_cnt = 0
        
        for step, input in enumerate(data_train):
            for idx, adm in enumerate(input): #########################
                seq_input = input[:idx+1]
                loss1_target = np.zeros((1, voc_size[2]))
                loss1_target[:, adm[2]] = 1     # ground truth
                loss3_target = np.full((1, voc_size[2]), -1) # (shape，fill_value)
                for idx, item in enumerate(adm[2]):
                    loss3_target[0][idx] = item

                target_output1, batch_neg_loss = model(seq_input)
                # target_output1= model(seq_input)

                loss1 = F.binary_cross_entropy_with_logits(target_output1, 
                                                           torch.FloatTensor(loss1_target).to(device))
                # with_logits会自动添加sigmoid计算loss
                loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), 
                                                 torch.LongTensor(loss3_target).to(device))
                # print('\nloss1:', loss1, '\nloss3:', loss3)
                """一次visit更新一次ddi rate，计算一次loss, 每个epoch输出loss均值"""


                target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]    
                # detach 参数不梯度更新，从计算图分离
                target_output1[target_output1 >= 0.5] = 1
                target_output1[target_output1 < 0.5] = 0
                y_label = np.where(target_output1 == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label]]) 
                
                w1 = max(1 - TARGET_DDI/(current_ddi_rate+1e-5), 0)
                loss = (1-w1) * (0.99 * loss1 + 0.01 * loss3) + w1 * batch_neg_loss

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                # print([x.grad for x in optimizer.param_groups[0]['params']])

                loss_record1.append(loss.item())

            llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % \
                    (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))


        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' 
                % (epoch, np.mean(loss_record1), elapsed_time, elapsed_time * (EPOCH - epoch - 1)/60))

        torch.save(model.state_dict(), 
                   open( os.path.join('saved', model_name, 
                                      'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
        print('')


if __name__ == '__main__':
    main()





