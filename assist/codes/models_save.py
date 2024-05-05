import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import json
from collections import defaultdict
import numpy as np
import math
import copy
import networkx as nx
import pickle
from torch.autograd import Variable
from data_loader import *

class KaNCD(nn.Module):

    def __init__(self, args, exer_n, student_n, knowledge_n, mf_type, dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(KaNCD, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)


    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb.weight
        exer_emb = self.exercise_emb.weight
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        stat_emb = stat_emb[stu_id]
        k_difficulty = k_difficulty[input_exercise]


        # prednet
        # Qe  (s-diff)  e_discrimination
        # pdb.set_trace()
        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def predict_proficiency_on_concepts(self):
        stu_emb = self.student_emb.weight
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        return stat_emb

    def predict_knowledge_embeddings(self):
        return self.knowledge_emb

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class OurModel(nn.Module):
    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim, StudentConceptTimes):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128
        super(OurModel, self).__init__()

        # prediction sub-net
        self.student_mu = nn.Embedding(self.student_n, 1)
        self.student_alpha = nn.Embedding(self.student_n, 1)
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        self.StudentMaskModule = StudentConceptTimes
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb.weight
        exer_emb = self.exercise_emb.weight
        student_mu = self.student_mu.weight
        student_alpha = self.student_alpha.weight
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = self.stat_full(stu_emb * knowledge_emb).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':
            k_difficulty = (exer_emb * knowledge_emb).sum(dim=-1, keepdim=False)  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = self.k_diff_full(exer_emb * knowledge_emb).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        # 0.00001 isdesigned to avoid torch.sum(self.StudentMaskModule, dim=1) = 0
        stat_emb_bias = torch.sum(self.StudentMaskModule * stat_emb, dim=1) / (torch.sum(self.StudentMaskModule, dim=1) + 0.00001)
        stat_emb_bias = torch.tanh(student_alpha * stat_emb_bias.reshape(-1,1) + student_mu)
        stat_emb = stat_emb + stat_emb_bias

        stat_emb = torch.sigmoid(stat_emb[stu_id])
        k_difficulty = torch.sigmoid(k_difficulty[input_exercise])

        input_x = input_knowledge_point* (stat_emb - k_difficulty) * e_discrimination
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

            

    def predict_proficiency_on_concepts(self):
        # before prednet
        stu_emb = self.student_emb.weight
        student_mu = self.student_mu.weight
        student_alpha = self.student_alpha.weight
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = self.stat_full(stu_emb * knowledge_emb).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)

        stat_emb_bias = torch.sum(self.StudentMaskModule * stat_emb, dim=1) / (torch.sum(self.StudentMaskModule, dim=1) + 0.00001)
        stat_emb_bias = torch.tanh(student_alpha * stat_emb_bias.reshape(-1,1) + student_mu)
        return torch.sigmoid(stat_emb + stat_emb_bias)

class NCDM(nn.Module):
    def __init__(self, args, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(NCDM, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stat_emb = self.student_emb.weight
        k_difficulty = self.k_difficulty.weight

        stat_emb = torch.sigmoid(stat_emb[stu_id])
        k_difficulty = torch.sigmoid(k_difficulty[input_exercise])

        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def predict_proficiency_on_concepts(self):
        stat_emb = torch.sigmoid(self.student_emb.weight)
        return stat_emb

class KSCD(nn.Module):
    def __init__(self, args, exer_n, student_n, knowledge_n, dim):
        self.knowledge_n = knowledge_n
        self.exer_count = exer_n
        self.stu_count = student_n
        self.lowdim = dim
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(KSCD, self).__init__()

        self.stu_emb = nn.Embedding(self.stu_count, self.lowdim)
        self.cpt_emb = nn.Embedding(self.knowledge_n, self.lowdim)
        self.exer_emb = nn.Embedding(self.exer_count, self.lowdim)


        self.prednet_full1 = PosLinear(self.knowledge_n + self.lowdim, self.knowledge_n, bias=False)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.knowledge_n + self.lowdim, self.knowledge_n, bias=False)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(1 * self.knowledge_n, 1)


    def forward(self, stu_id, exer_id, k_ids):
        stu_emb = self.stu_emb.weight
        exer_emb = self.exer_emb.weight
        exer_q_mat = k_ids
        stu_ability = torch.mm(stu_emb, self.cpt_emb.weight.T).sigmoid()
        exer_diff = torch.mm(exer_emb, self.cpt_emb.weight.T).sigmoid()

        stu_ability = stu_ability[stu_id]
        exer_diff = exer_diff[exer_id]
        batch_stu_vector = stu_ability.repeat(1, self.knowledge_n).reshape(stu_ability.shape[0], self.knowledge_n,
                                                                         stu_ability.shape[1])
        batch_exer_vector = exer_diff.repeat(1, self.knowledge_n).reshape(exer_diff.shape[0], self.knowledge_n,
                                                                        exer_diff.shape[1])
        kn_vector = self.cpt_emb.weight.repeat(stu_ability.shape[0], 1).reshape(stu_ability.shape[0], self.knowledge_n,
                                                                                self.lowdim)
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))
        sum_out = torch.sum(o * exer_q_mat.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(exer_q_mat, dim=1).unsqueeze(1)
        y_pd = sum_out / count_of_concept
        return y_pd.view(-1)

    def predict_proficiency_on_concepts(self):
        stu_emb = self.stu_emb.weight
        stu_ability = torch.mm(stu_emb, self.cpt_emb.weight.T).sigmoid()
        return stu_ability

class alation1(nn.Module):
    def __init__(self, args, knowledge_n, exer_n, student_n, StudentConceptTimes):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(alation1, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        self.student_mu = nn.Embedding(self.emb_num, 1)
        self.student_alpha = nn.Embedding(self.emb_num, 1)
        self.StudentMaskModule = StudentConceptTimes
        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stat_emb = self.student_emb.weight
        k_difficulty = self.k_difficulty.weight
        student_mu = self.student_mu.weight
        student_alpha = self.student_alpha.weight

        stat_emb_bias = torch.sum(self.StudentMaskModule * stat_emb, dim=1) / (
                    torch.sum(self.StudentMaskModule, dim=1) + 0.00001)
        stat_emb_bias = torch.tanh(student_alpha * stat_emb_bias.reshape(-1, 1) + student_mu)
        stat_emb = stat_emb + stat_emb_bias

        stat_emb = torch.sigmoid(stat_emb[stu_id])
        k_difficulty = torch.sigmoid(k_difficulty[input_exercise])


        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def predict_proficiency_on_concepts(self):
        stat_emb = self.student_emb.weight
        student_mu = self.student_mu.weight
        student_alpha = self.student_alpha.weight
        stat_emb_bias = torch.sum(self.StudentMaskModule * stat_emb, dim=1) / (
                    torch.sum(self.StudentMaskModule, dim=1) + 0.00001)
        stat_emb_bias = torch.tanh(student_alpha * stat_emb_bias.reshape(-1, 1) + student_mu)
        stat_emb = stat_emb + stat_emb_bias
        stat_emb = torch.sigmoid(stat_emb)
        return stat_emb

class alation2(nn.Module):
    def __init__(self, args, exer_n, student_n, knowledge_n, mf_type, dim, StudentConceptTimes):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(alation2, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.student_mu = nn.Embedding(self.student_n, 1)
        self.student_alpha = nn.Embedding(self.student_n, 1)
        self.StudentMaskModule = StudentConceptTimes

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)


    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb.weight
        exer_emb = self.exercise_emb.weight
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = (self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        student_mu = self.student_mu.weight
        student_alpha = self.student_alpha.weight

        stat_emb_bias = torch.sum(self.StudentMaskModule * stat_emb, dim=1) / (
                    torch.sum(self.StudentMaskModule, dim=1) + 0.00001)
        stat_emb_bias = torch.tanh(student_alpha * stat_emb_bias.reshape(-1, 1) + student_mu)
        stat_emb = torch.sigmoid(stat_emb + stat_emb_bias)

        stat_emb = stat_emb[stu_id]
        k_difficulty = k_difficulty[input_exercise]

        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def predict_proficiency_on_concepts(self):
        stu_emb = self.student_emb.weight
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = (self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        
        student_mu = self.student_mu.weight
        student_alpha = self.student_alpha.weight

        stat_emb_bias = torch.sum(self.StudentMaskModule * stat_emb, dim=1) / (
                    torch.sum(self.StudentMaskModule, dim=1) + 0.00001)
        stat_emb_bias = torch.tanh(student_alpha * stat_emb_bias.reshape(-1, 1) + student_mu)
        stat_emb = torch.sigmoid(stat_emb + stat_emb_bias)
        return stat_emb


class altion3(nn.Module):
    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim, StudentConceptTimes):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128
        super(altion3, self).__init__()

        # prediction sub-net
        self.student_mu = nn.Embedding(self.student_n, 1)
        self.student_alpha = nn.Embedding(self.student_n, 1)

        self.student_emb = nn.Embedding(self.student_n, self.knowledge_n)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_n)

        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.StudentMaskModule = StudentConceptTimes
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stat_emb = self.student_emb.weight
        k_difficulty = self.exercise_emb.weight
        student_mu = self.student_mu.weight
        student_alpha = self.student_alpha.weight

        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        # 0.00001 isdesigned to avoid torch.sum(self.StudentMaskModule, dim=1) = 0
        stat_emb_bias = torch.sum(self.StudentMaskModule * stat_emb, dim=1) / (
                    torch.sum(self.StudentMaskModule, dim=1) + 0.00001)
        stat_emb_bias = torch.tanh(student_alpha * stat_emb_bias.reshape(-1, 1) + student_mu)
        stat_emb = stat_emb + stat_emb_bias

        stat_emb = torch.sigmoid(stat_emb[stu_id])
        k_difficulty = torch.sigmoid(k_difficulty[input_exercise])

        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def predict_proficiency_on_concepts(self):
        # before prednet
        stat_emb = self.student_emb.weight
        student_mu = self.student_mu.weight
        student_alpha = self.student_alpha.weight
        # 0.00001 isdesigned to avoid torch.sum(self.StudentMaskModule, dim=1) = 0
        stat_emb_bias = torch.sum(self.StudentMaskModule * stat_emb, dim=1) / (
                    torch.sum(self.StudentMaskModule, dim=1) + 0.00001)
        stat_emb_bias = torch.tanh(student_alpha * stat_emb_bias.reshape(-1, 1) + student_mu)
        return torch.sigmoid(stat_emb + stat_emb_bias)
