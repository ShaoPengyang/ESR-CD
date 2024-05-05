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

from collections import defaultdict
from torch.autograd import Variable


train_data_json = '../data/ASSIST/train_set.json'
eval_data_json = '../data/ASSIST/eval_set.json'
test_data_json = '../data/ASSIST/test_set.json'
# train_data_json = '../data/ASSIST/train_set_ood.json'
# eval_data_json = '../data/ASSIST/eval_set_ood.json'
# test_data_json = '../data/ASSIST/test_set_ood.json'

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
        with open(self.data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        self.config_file = 'config.txt'
        with open(self.config_file) as i_f:
            i_f.readline()
            student_n, exercise_n, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.student_dim = int(student_n)
        self.exercise_dim = int(exercise_n)

    def load_data(self, return_matrix = False):
        '''
        if first load, use comment part.
        :return:
        '''
        self.dataset = []
        self.k_ids = []
        self.Q_matrix = torch.zeros((self.exercise_dim,self.knowledge_dim))
        for idx, log in enumerate(self.data):
            u_id = log['user_id'] - 1
            e_id = log['exer_id'] - 1
            y = log['score']
            if y > 1 or y < 0:
                pdb.set_trace()
            self.dataset.append([u_id, e_id, y])
            for knowledge_code in log['knowledge_code']:
                self.Q_matrix[e_id][knowledge_code - 1] = 1.0

        # self.Q_matrix = torch.cuda.LongTensor(self.k_ids)
        self.data_len = len(self.dataset)

        self.new_Q_matrix = torch.zeros((self.exercise_dim,self.knowledge_dim))
        if return_matrix == True:
            return self.Q_matrix.to_sparse()

    def update_Q(self, input_Q):
        self.new_Q_matrix = input_Q

    def __len__(self):
        return  self.data_len

    def __getitem__(self, idx):
        u_id = self.dataset[idx][0]
        i_id = self.dataset[idx][1]
        label = self.dataset[idx][2]
        k_id = self.Q_matrix[i_id]
        new_k_id = self.new_Q_matrix[i_id]
        return u_id, i_id, k_id,new_k_id, label


def generate_random_sample(DeficientConceptDict, ConceptMapExercise, ExerciseMapConcept, max_number=10):
    '''
    :param DeficientConceptDict: where needs to perform data augmentation
    :param ConceptMapExercise: concept:{exercise1, exercise2, ... , exercise S}
    :param max_number: After sampling, how many times we allow {student, concept} pair to have? (part will be deleted in augmentation)
    :return: random_sample (candidates)
    '''
    random_sample = []
    corresponding_concept_vector = []
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exercise_n, knowledge_n = i_f.readline().split(',')
    student_n, exercise_n, knowledge_n = int(student_n), int(exercise_n), int(knowledge_n)

    for student, interactions in DeficientConceptDict.items():
        for concept, exercises in interactions.items():
            all_exercises_set = ConceptMapExercise[concept]
            done_exercises_set = set(exercises)
            differences = np.array(list(all_exercises_set - done_exercises_set))
            if len(differences) > (max_number-len(done_exercises_set)):
                try:
                    add_part = np.random.choice(differences, size=max_number-len(done_exercises_set), replace=False)
                except:
                    pdb.set_trace()
            else:
                add_part = differences
        # pdb.set_trace()
        assert len(add_part) == len(set(list(add_part))), "repeatable elements!!!"
        for exercise in add_part:
            knowledge_emb = [0.] * knowledge_n
            for knowledge_code in ExerciseMapConcept[exercise]:
                knowledge_emb[knowledge_code] = 1.0

            random_sample.append([student, exercise])
            corresponding_concept_vector.append(knowledge_emb)
    random_sample = np.array(random_sample)

    return random_sample, corresponding_concept_vector


def obtain_adjency_matrix(args):
    with open(train_data_json, encoding='utf8') as i_f:
        data = json.load(i_f)
    train_data_user_score1,train_data_user_score0 = defaultdict(set), defaultdict(set)
    train_data_item_score1,train_data_item_score0 = defaultdict(set), defaultdict(set)
    for idx, log in enumerate(data):
        u_id = log['user_id'] - 1
        i_id = log['exer_id'] - 1
        if log['score'] == 1:
            train_data_user_score1[u_id].add(int(i_id))
            train_data_item_score1[int(i_id)].add(u_id)
        elif log['score'] == 0:
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
    with open(train_data_json, encoding='utf8') as i_f:
        data = json.load(i_f)
    train_data_user_score1 = defaultdict(set)
    train_data_item_score1 = defaultdict(set)
    for idx, log in enumerate(data):
        u_id = log['user_id'] - 1
        i_id = log['exer_id'] - 1
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


def Motif_Generated(user_num=1, exer_num=1, concept_num = 1):
    with open(train_data_json, encoding='utf8') as i_f:
        data = json.load(i_f)
    StudentExerciseMatrix1, StudentExerciseMatrix2 = np.zeros((user_num,exer_num)), np.zeros((user_num,exer_num))
    StuentConceptTimes = np.zeros((user_num, concept_num))
    for idx, log in enumerate(data):
        u_id = log['user_id'] - 1
        e_id = log['exer_id'] - 1
        label = log['score']
        for knowledge_id in log['knowledge_code']:
            StuentConceptTimes[u_id][knowledge_id-1] += 1
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



