import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import networkx as nx
import json
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
from data_loader import *
from models import *
from utils import *
from torch.utils.data import DataLoader
import time
import math
import pdb

def train(args):
    train_dataset = EduData(type='train')
    sparse_Q = train_dataset.load_data(return_matrix=True)
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)
    test_dataset = EduData(type='predict')
    test_dataset.load_data()
    test_loader = DataLoader(test_dataset, batch_size=8192, shuffle=False)
    eval_dataset = EduData(type='eval')
    eval_dataset.load_data()
    eval_loader = DataLoader(eval_dataset, batch_size=8192, shuffle=False)
    print("len of training dataset is: " + str(len(train_dataset)))
    print("len of eval dataset is: " + str(len(eval_dataset)))
    print("len of test dataset is: " + str(len(test_dataset)))
    loss_function = nn.BCELoss(reduction='mean')

    hyper_e_1,hyper_e_2, record_matrix = Motif_Generated(args.student_n,args.exer_n, args.knowledge_n)


    '''
    update Q
    '''
    print(sparse_Q.to_dense().sum())
    hyperpara1, hyper_threshold = 1, 0.5

    hyper_e_1_sum = hyper_e_1.sum(dim=1, keepdim=True) + 0.00001
    hyper_e_2_sum = hyper_e_2.sum(dim=1, keepdim=True) + 0.00001
    input_Q1 = torch.sparse.mm(sparse_Q.t(), hyper_e_1.t()).t() / hyper_e_1_sum
    input_Q2 = torch.sparse.mm(sparse_Q.t(), hyper_e_2.t()).t() / hyper_e_2_sum
    input_Q = hyperpara1*(input_Q1+input_Q2)
    input_Q[input_Q <= hyper_threshold] = 0

    input_Q = input_Q + sparse_Q.to_dense()
    input_Q[input_Q>0] = 1
    print(input_Q.sum())
    train_dataset.update_Q(input_Q)
    test_dataset.update_Q(input_Q)
    eval_dataset.update_Q(input_Q)
    net =altion3(args.exer_n, args.student_n, args.knowledge_n, 'gmf', 64, torch.cuda.LongTensor(record_matrix))
    # net = OurModel(args.exer_n, args.student_n, args.knowledge_n, 'gmf', 64, torch.cuda.LongTensor(record_matrix))
    net = net.cuda()
    optimizer_net = optim.Adam(net.parameters(), lr=0.005)

    for epoch in range(50):
        net.train()
        running_loss = []
        for idx, (input_stu_ids, input_exer_ids, kid1, kid2, labels) in enumerate(train_loader):
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            kid2 = kid2.cuda().long()
            labels = labels.cuda().float()
            optimizer_net.zero_grad()
            output = net.forward(input_stu_ids, input_exer_ids, kid2)
            try:
                edu_loss = loss_function(output, labels.float())
            except:
                pdb.set_trace()
            edu_loss.backward()
            optimizer_net.step()
            running_loss.append(edu_loss.item())

        print("eval results:")
        predict(net, eval_loader, epoch)
        print("test results:")
        predict2(net, test_loader, epoch)

def predict(net, test_loader, epoch):
    net.eval()
    with torch.no_grad():
        correct_count, exer_count = 0, 0
        pred_all, label_all = [], []
        for input_stu_ids, input_exer_ids, _, input_knowledge_embs, labels in test_loader:
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            labels = labels.cuda()
            input_knowledge_embs = input_knowledge_embs.cuda()
            output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output = output.view(-1)
            correct_count += ((output >= 0.5) == labels).sum()
            exer_count += len(labels)
            pred_all += output.to(torch.device('cpu')).tolist()
            label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    # print('[epoch= %d], accuracy= %f, rmse= %f, auc= %f' % (epoch + 1, accuracy, rmse, auc))
    print('[epoch= %d], accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))

def predict2(net, test_loader, epoch):
    net.eval()
    users, items, know_ids, predicted_scores, predicted_thetas = [], [], [], [], []
    with torch.no_grad():
        prof = net.predict_proficiency_on_concepts().to(torch.device('cpu')).numpy()
        # print(np.mean(prof))
        correct_count, exer_count = 0, 0
        pred_all, label_all = [], []
        for input_stu_ids, input_exer_ids, input_knowledge_embs1, input_knowledge_embs2, labels in test_loader:
            for _ in input_stu_ids.cpu().numpy():
                predicted_thetas.append(list(prof[_]))
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            labels = labels.cuda()
            input_knowledge_embs1 = input_knowledge_embs1.cuda()
            input_knowledge_embs2 = input_knowledge_embs2.cuda()
            users.extend(input_stu_ids.to(torch.device('cpu')).tolist())
            items.extend(input_exer_ids.to(torch.device('cpu')).tolist())
            know_ids.extend(input_knowledge_embs1.to(torch.device('cpu')).tolist())
            output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs2)
            output = output.view(-1)
            predicted_scores.extend(labels.to(torch.device('cpu')).tolist())
            correct_count += ((output >= 0.5) == labels).sum()
            exer_count += len(labels)
            pred_all += output.to(torch.device('cpu')).tolist()
            label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    # print('[epoch= %d], accuracy= %f, rmse= %f, auc= %f' % (epoch + 1, accuracy, rmse, auc))

    DOA =  doa_report(users, items, know_ids, predicted_scores, predicted_thetas)
    print('[epoch= %d], accuracy= %f, rmse= %f, auc= %f doa= %f' % (epoch+1, accuracy, rmse, auc, DOA['doa']))

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    train(args)
