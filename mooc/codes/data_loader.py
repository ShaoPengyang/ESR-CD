import json
import torch
import math
import random
import pickle
import time
import pdb
import copy
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd

from collections import defaultdict
from torch.autograd import Variable

# train_data_json = '../data/coarse/train_set.npy'
# eval_data_json = '../data/coarse/eval_set.npy'
# test_data_json = '../data/coarse/test_set.npy'
train_data_json = '../data/coarse/train_set_ood.npy'
eval_data_json = '../data/coarse/eval_set_ood.npy'
test_data_json = '../data/coarse/test_set_ood.npy'
item2knowledge_path = "../data/coarse/item2knowledge.npy"
item2knowledge = np.load(item2knowledge_path, allow_pickle = True).item()

def obtain_adjency_matrix(args):
    data = np.load(train_data_json, allow_pickle=True)
    train_data_user_score1,train_data_user_score0 = defaultdict(set), defaultdict(set)
    train_data_item_score1,train_data_item_score0 = defaultdict(set), defaultdict(set)
    for idx, log in enumerate(data):
        u_id = log[0] -1
        i_id = log[1] -1
        if log[2] == 1:
            train_data_user_score1[u_id].add(int(i_id))
            train_data_item_score1[int(i_id)].add(u_id)
        elif log[2] == 0:
            train_data_user_score0[u_id].add(int(i_id))
            train_data_item_score0[int(i_id)].add(u_id)
        else:
            assert False, 'rating must be 1 or 0.'

    u_d_1 = readD(args, train_data_user_score1, args.student_n)
    i_d_1 = readD(args, train_data_item_score1, args.exer_n)
    u_d_0 = readD(args, train_data_user_score0, args.student_n)
    i_d_0 = readD(args, train_data_item_score0, args.exer_n)
    sparse_u_i_1 = readTrainSparseMatrix(args, train_data_user_score1,u_d_1, i_d_1,  True)
    sparse_i_u_1 = readTrainSparseMatrix(args, train_data_item_score1,u_d_1, i_d_1, False)
    sparse_u_i_0 = readTrainSparseMatrix(args, train_data_user_score0,u_d_0, i_d_0, True)
    sparse_i_u_0 = readTrainSparseMatrix(args, train_data_item_score0,u_d_0, i_d_0, False)


    return [u_d_1,i_d_1,sparse_u_i_1,sparse_i_u_1], [u_d_0, i_d_0, sparse_u_i_0, sparse_i_u_0]

def obtain_adjency_matrix2(args):
    data = np.load(train_data_json, allow_pickle=True)
    train_data_user_score1 = defaultdict(set)
    train_data_item_score1 = defaultdict(set)
    for idx, log in enumerate(data):
        u_id = log[0] -1
        i_id = log[1] -1
        train_data_user_score1[u_id].add(int(i_id))
        train_data_item_score1[int(i_id)].add(u_id)
    u_d_1 = readD(args, train_data_user_score1, args.student_n)
    i_d_1 = readD(args, train_data_item_score1, args.exer_n)
    sparse_u_i_1 = readTrainSparseMatrix(args, train_data_user_score1,u_d_1, i_d_1,  True)
    sparse_i_u_1 = readTrainSparseMatrix(args, train_data_item_score1,u_d_1, i_d_1, False)
    return u_d_1,i_d_1,sparse_u_i_1,sparse_i_u_1 



def readD(args, set_matrix,num_):
    user_d=[]
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)
        user_d.append(len_set)
    return user_d


def readTrainSparseMatrix(args, set_matrix,u_d,i_d, is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    exer_num = args.exer_n
    student_n = args.student_n
    if is_user:
        d_i=u_d
        d_j=i_d
        user_items_matrix_i.append([student_n-1, exer_num-1])
        user_items_matrix_v.append(0)
    else:
        d_i=i_d
        d_j=u_d
        user_items_matrix_i.append([exer_num - 1, student_n - 1])
        user_items_matrix_v.append(0)
    for i in set_matrix:
        len_set=len(set_matrix[i])
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            user_items_matrix_v.append(d_i_j)
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

class EduData(data.Dataset):
    def __init__(self, type='train'):
        super(EduData, self).__init__()
        if type == 'train':
            self.data_file = train_data_json
            self.type = 'train'
        elif type == 'predict':
            self.data_file = test_data_json
            self.type = 'predict'
        elif type == 'eval':
            self.data_file = eval_data_json
            self.type = 'eval'
        else:
            assert False, 'type can only be selected from train or predict'

        self.data = np.load(self.data_file, allow_pickle=True)
        self.config_file = 'config.txt'
        with open(self.config_file) as i_f:
            i_f.readline()
            student_n, exercise_n, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.student_dim = int(student_n)
        self.exercise_dim = int(exercise_n)

    def load_data(self, return_matrix=False):
        self.data_len = self.data.shape[0]
        self.Q_matrix = torch.zeros((self.exercise_dim,self.knowledge_dim))
        self.new_Q_matrix = torch.zeros((self.exercise_dim,self.knowledge_dim))
        for index in range(len(self.data)):
            self.Q_matrix[self.data[index][1]-1, item2knowledge[self.data[index][1]]-1] = 1

        if return_matrix == True:
            return self.Q_matrix.to_sparse()


    def update_Q(self, input_Q):
        self.new_Q_matrix = input_Q

    def __len__(self):
        return  self.data_len

    def __getitem__(self, idx):
        u_id = self.data[idx][0] - 1
        i_id = self.data[idx][1] - 1
        label = self.data[idx][2]
        k_id = self.Q_matrix[i_id]
        new_k_id = self.new_Q_matrix[i_id]
        return u_id, i_id, k_id, new_k_id, label

def Motif_Generated(user_num=1, exer_num=1, concept_num = 1):
    data = np.load(train_data_json, allow_pickle=True)
    StudentExerciseMatrix1, StudentExerciseMatrix2 = np.zeros((user_num,exer_num)), np.zeros((user_num,exer_num))
    StuentConceptTimes = np.zeros((user_num, concept_num))
    for idx, log in enumerate(data):
        u_id = log[0] - 1
        e_id = log[1] - 1

        label = log[2]
        StuentConceptTimes[u_id][item2knowledge[log[1]]-1] += 1
        if label == 1:
            StudentExerciseMatrix1[u_id][e_id] = 1
        if label == 0:
            StudentExerciseMatrix2[u_id][e_id] = 1

    lts_data_tensor = torch.FloatTensor(StudentExerciseMatrix1)
    cos_sim1 = sim_matrix(lts_data_tensor.t(),lts_data_tensor.t())
    lts_data_tensor = torch.FloatTensor(StudentExerciseMatrix2)
    cos_sim2 = sim_matrix(lts_data_tensor.t(),lts_data_tensor.t())

    # An operation.
    for i in range(cos_sim1.shape[0]):
        cos_sim1[i][i] = 0
        cos_sim2[i][i] = 0
    StuentConceptTimes[StuentConceptTimes>0] = 1
    return cos_sim1, cos_sim2, StuentConceptTimes

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    a_norm = a_norm.to_sparse()
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
